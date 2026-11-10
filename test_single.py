import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from generate_attention_results import create_original_overlays


ROOT = Path(__file__).resolve().parent
OCCLUSION_RATIOS = (0.1, 0.2, 0.3, 0.5)
RESULT_FIELDS = (
    "model",
    "variant",
    "mode",
    "occlusion_ratio_percent",
    "mAP_percent",
    "rank1_percent",
    "rank5_percent",
    "checkpoint",
)
CONDITION_PATTERN = re.compile(
    r"Evaluation condition:\s*([^;]+);\s*occlusion_ratio=([0-9.]+)"
)
METRIC_PATTERNS = {
    "mAP_percent": re.compile(r"\bmAP:\s*([0-9]+(?:\.[0-9]+)?)%"),
    "rank1_percent": re.compile(
        r"CMC curve,\s*Rank-1\s*:\s*([0-9]+(?:\.[0-9]+)?)%"
    ),
    "rank5_percent": re.compile(
        r"CMC curve,\s*Rank-5\s*:\s*([0-9]+(?:\.[0-9]+)?)%"
    ),
}


def _natural_sort_key(value):
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _checkpoints_in_run(run_dir, model):
    prefix = model + "_"
    checkpoints = []
    for checkpoint in run_dir.glob(prefix + "*/" + prefix + "*_final.pth"):
        case_name = checkpoint.parent.name
        if checkpoint.name != case_name + "_final.pth":
            continue
        variant = case_name[len(prefix):]
        checkpoints.append((case_name, variant, checkpoint))
    return sorted(checkpoints, key=lambda item: _natural_sort_key(item[0]))


def find_checkpoints(checkpoint_root, model, run_id=None):
    if run_id:
        run_dirs = [checkpoint_root / run_id]
    else:
        run_dirs = sorted(
            (path for path in checkpoint_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        checkpoints = _checkpoints_in_run(run_dir, model)
        if checkpoints:
            return run_dir.name, checkpoints

    if run_id:
        raise FileNotFoundError(
            "Run {} does not contain any {} *_final.pth checkpoints under "
            "{}_* case folders.".format(run_id, model.upper(), model)
        )
    raise FileNotFoundError(
        "No run under {} contains {} *_final.pth checkpoints under {}_* case "
        "folders.".format(checkpoint_root, model.upper(), model)
    )


def read_test_metrics(log_path):
    results = []
    current = None
    for line in log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        condition_match = CONDITION_PATTERN.search(line)
        if condition_match:
            if current is not None:
                results.append(current)
            current = {
                "mode": condition_match.group(1).strip(),
                "occlusion_ratio_percent": (
                    float(condition_match.group(2)) * 100
                ),
            }
            continue
        if current is not None:
            for name, pattern in METRIC_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    current[name] = float(match.group(1))

    if current is not None:
        results.append(current)

    expected_ratios = (0.0,) + tuple(ratio * 100 for ratio in OCCLUSION_RATIOS)
    actual_ratios = [row["occlusion_ratio_percent"] for row in results]
    if actual_ratios != list(expected_ratios):
        raise RuntimeError(
            "Expected evaluation ratios {}, found {} in {}.".format(
                list(expected_ratios), actual_ratios, log_path
            )
        )
    for result in results:
        missing = set(METRIC_PATTERNS).difference(result)
        if missing:
            raise RuntimeError(
                "Missing test metrics {} in {}".format(
                    sorted(missing), log_path
                )
            )
    return results


def write_metrics_csv(metrics_path, results):
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def _attention_keys(condition_dir):
    index_path = condition_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError("Attention index not found: {}".format(index_path))
    with index_path.open("r", encoding="utf-8") as index_file:
        return {
            json.loads(line)["key"]
            for line in index_file
            if line.strip()
        }


def run_test(
    case_name,
    model,
    variant,
    checkpoint,
    config_path,
    output_root,
    num_images,
    alpha,
    env,
):
    case_output = output_root / "logs" / case_name
    case_output.mkdir(parents=True, exist_ok=False)
    map_root = output_root / "maps" / case_name
    runner_log = case_output / "runner.log"
    test_log_path = case_output / "test_log.txt"

    command = [
        sys.executable,
        str(ROOT / "test_clipreid.py"),
        "--config_file",
        config_path,
        "TEST.WEIGHT",
        str(checkpoint),
        "TEST.ATTENTION.ENABLED",
        "True",
        "TEST.ATTENTION.NUM_IMAGES",
        str(num_images),
        "TEST.ATTENTION.MAP_DIR",
        str(map_root),
        "TEST.OCCLUSION.ENABLED",
        "True",
        "TEST.OCCLUSION.RATIOS",
        str(list(OCCLUSION_RATIOS)),
        "OUTPUT_DIR",
        str(case_output),
    ]
    print("[TEST] {} -> {}".format(case_name, checkpoint))
    with runner_log.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        tail = runner_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-40:]
        raise RuntimeError(
            "Test failed for {} with exit code {}. Log: {}\n{}".format(
                case_name,
                completed.returncode,
                runner_log,
                "\n".join(tail),
            )
        )

    metrics = read_test_metrics(test_log_path)
    comparison_root = output_root / "comparisons"
    reference_keys = None
    for ratio in (0.0,) + OCCLUSION_RATIOS:
        condition_name = "occlusion_{:03d}".format(round(ratio * 100))
        condition_dirs = [
            path
            for path in map_root.glob("**/" + condition_name)
            if path.parent.name == checkpoint.stem
        ]
        if len(condition_dirs) != 1:
            raise RuntimeError(
                "Expected one {} map directory for {}, found {}.".format(
                    condition_name, case_name, len(condition_dirs)
                )
            )
        condition_dir = condition_dirs[0]
        keys = _attention_keys(condition_dir)
        if len(keys) != num_images:
            raise RuntimeError(
                "Expected attention maps for {} images in {}, found {}.".format(
                    num_images, condition_dir, len(keys)
                )
            )
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise RuntimeError(
                "The preview image set changed between occlusion conditions "
                "for {}.".format(case_name)
            )
        image_count, overlay_dir = create_original_overlays(
            condition_dir, comparison_root, case_name, alpha
        )
        if image_count != num_images:
            raise RuntimeError(
                "Expected {} overlays for {}, found {}.".format(
                    num_images, case_name, image_count
                )
            )
        print(
            "[OVERLAY] {} {}: {} images -> {}".format(
                case_name, condition_name, image_count, overlay_dir
            )
        )

    return (
        [
            {
                "model": model,
                "variant": variant,
                "checkpoint": str(checkpoint),
                **result,
            }
            for result in metrics
        ],
        reference_keys,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every selected model checkpoint on the full test set, "
            "saving attention previews for 10 shared images."
        )
    )
    parser.add_argument(
        "--model",
        choices=("cnn", "vit"),
        default="cnn",
        help="Model family to evaluate; ViT is available but not run by default.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="outputs",
        help="Root containing run-id/model-variant/model-variant_final.pth.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Use this output run; otherwise select the newest run with matching checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/test_single",
        help="Root directory for logs, metrics, attention maps, and previews.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of shared images to save attention previews for.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Attention heatmap blend strength from 0 to 1.",
    )
    args = parser.parse_args()

    if args.num_images < 2:
        parser.error("--num-images must be at least 2.")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1.")

    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = ROOT / checkpoint_root
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            "Checkpoint folder not found: {}".format(checkpoint_root)
        )
    run_id, checkpoints = find_checkpoints(
        checkpoint_root, args.model, args.run_id or None
    )

    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = ROOT / output_base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = output_base / "{}_{}_{}".format(
        args.model, run_id, timestamp
    )
    output_root.mkdir(parents=True, exist_ok=False)

    config_path = "configs/person/{}_clipreid.yml".format(args.model)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    all_results = []
    preview_keys = None
    for case_name, variant, checkpoint in checkpoints:
        variant_results, variant_preview_keys = run_test(
            case_name,
            args.model,
            variant,
            checkpoint,
            config_path,
            output_root,
            args.num_images,
            args.alpha,
            env,
        )
        if preview_keys is None:
            preview_keys = variant_preview_keys
        elif variant_preview_keys != preview_keys:
            raise RuntimeError(
                "The preview image set changed between model variants."
            )
        all_results.extend(variant_results)

    metrics_path = output_root / "metrics.csv"
    write_metrics_csv(metrics_path, all_results)
    print("[DONE] Metrics: {}".format(metrics_path))
    print("[DONE] Image comparisons: {}".format(output_root / "comparisons"))


if __name__ == "__main__":
    main()