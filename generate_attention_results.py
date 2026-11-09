import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from utils.attention_maps import colorize_attention_overlay


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "cnn_baseline": "configs/person/cnn_clipreid.yml",
    "cnn_random_k": "configs/person/cnn_clipreid.yml",
    "cnn_top_k": "configs/person/cnn_clipreid.yml",
    "cnn_top_k_random_pool": "configs/person/cnn_clipreid.yml",
    "vit_baseline": "configs/person/vit_clipreid.yml",
    "vit_random_k": "configs/person/vit_clipreid.yml",
    "vit_top_k": "configs/person/vit_clipreid.yml",
    "vit_top_k_random_pool": "configs/person/vit_clipreid.yml",
}
DEFAULT_RATIOS = ()


def _resolve_from_root(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def find_checkpoints(checkpoint_root, case_names):
    checkpoints = {}
    missing = []

    for case_name in case_names:
        expected_name = case_name + "_final.pth"
        matches = list(checkpoint_root.glob("*/{}/{}".format(case_name, expected_name)))
        if not matches:
            missing.append(case_name)
            continue
        checkpoints[case_name] = max(matches, key=lambda path: path.stat().st_mtime)

    return checkpoints, missing


def create_original_overlays(condition_dir, comparison_root, case_name, alpha):
    index_path = condition_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError("Attention index not found: {}".format(index_path))

    records = {}
    with index_path.open("r", encoding="utf-8") as index_file:
        for line in index_file:
            record = json.loads(line)
            records[record["key"]] = record

    if not records:
        raise RuntimeError("Attention index is empty: {}".format(index_path))

    originals_dir = comparison_root / "originals"
    overlays_dir = comparison_root / "overlays" / case_name / condition_dir.name
    originals_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    for key, record in records.items():
        original_path = condition_dir / record["original"]
        attention_path = condition_dir / record["attention"]
        if not original_path.is_file():
            raise FileNotFoundError("Original image not found: {}".format(original_path))
        if not attention_path.is_file():
            raise FileNotFoundError("Attention map not found: {}".format(attention_path))

        with Image.open(original_path) as source_file:
            original = source_file.convert("RGB")
        with Image.open(attention_path) as map_file:
            attention = map_file.convert("L").resize(
                original.size, Image.Resampling.BILINEAR
            )
        original.save(originals_dir / (key + ".png"), format="PNG")
        colorize_attention_overlay(original, attention, alpha).save(
            overlays_dir / (key + "_attention.png"), format="PNG"
        )

    return len(records), overlays_dir


def _run_evaluation(
    case_name, checkpoint, config_path, map_root, log_root, ratios, num_images
):
    checkpoint_arg = checkpoint.relative_to(ROOT).as_posix()
    map_root_arg = map_root.relative_to(ROOT).as_posix()
    log_dir = log_root / case_name
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir_arg = log_dir.relative_to(ROOT).as_posix()

    command = [
        sys.executable,
        str(ROOT / "test_clipreid.py"),
        "--config_file",
        config_path,
        "TEST.WEIGHT",
        checkpoint_arg,
        "TEST.ATTENTION.ENABLED",
        "True",
        "TEST.ATTENTION.NUM_IMAGES",
        str(num_images),
        "TEST.ATTENTION.MAP_DIR",
        map_root_arg,
        "TEST.OCCLUSION.ENABLED",
        "True",
        "TEST.OCCLUSION.RATIOS",
        str(list(ratios)),
        "OUTPUT_DIR",
        output_dir_arg,
    ]
    print("[TEST]", case_name, "->", checkpoint)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create attention overlays for the same small test-image subset "
            "across available final checkpoints."
        )
    )
    parser.add_argument(
        "--checkpoint-root",
        default="outputs",
        help="Folder containing run-id/case-name/case-name_final.pth.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/attention_overlays",
        help="Folder for attention maps, comparison images, and test logs.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of shared query/gallery images to test per model (default: 10).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=tuple(EXPERIMENTS),
        help="Experiment names to process; defaults to all eight.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=DEFAULT_RATIOS,
        help="Occlusion ratios to evaluate; baseline (0%%) is always included.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Attention heatmap blend strength from 0 to 1.",
    )
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1.")
    if args.num_images < 2:
        parser.error("--num-images must be at least 2.")
    if any(not 0 < ratio <= 1 for ratio in args.ratios):
        parser.error("Each --ratios value must be greater than 0 and at most 1.")
    if len(set(args.ratios)) != len(args.ratios):
        parser.error("--ratios values must not contain duplicates.")

    checkpoint_root = _resolve_from_root(args.checkpoint_root)
    output_root = _resolve_from_root(args.output_dir)
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            "Checkpoint folder not found: {}".format(checkpoint_root)
        )

    checkpoints, missing = find_checkpoints(checkpoint_root, args.cases)
    if not checkpoints:
        raise FileNotFoundError(
            "No matching final checkpoints found under {}.".format(checkpoint_root)
        )
    if missing:
        print(
            "[WARN] Missing checkpoints for: {}. These cases will be skipped.".format(
                ", ".join(missing)
            )
        )

    map_root = output_root / "maps"
    log_root = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_root = output_root / "comparisons"

    for case_name, checkpoint in checkpoints.items():
        case_map_root = map_root / case_name
        _run_evaluation(
            case_name,
            checkpoint,
            EXPERIMENTS[case_name],
            case_map_root,
            log_root,
            args.ratios,
            args.num_images,
        )

        condition_root = case_map_root
        ratios = (0.0,) + tuple(args.ratios)
        for ratio in ratios:
            condition_name = "occlusion_{:03d}".format(round(ratio * 100))
            matching_dirs = [
                path
                for path in condition_root.glob("**/" + condition_name)
                if path.parent.name == checkpoint.stem
            ]
            if len(matching_dirs) != 1:
                raise RuntimeError(
                    "Expected one {} attention folder for {}, found {}.".format(
                        condition_name, case_name, len(matching_dirs)
                    )
                )
            image_count, overlay_dir = create_original_overlays(
                matching_dirs[0], comparison_root, case_name, args.alpha
            )
            print(
                "[OVERLAY] {} {}: {} images -> {}".format(
                    case_name, condition_name, image_count, overlay_dir
                )
            )

    print("[DONE] Results saved under {}".format(output_root))
    if missing:
        print("[WARN] Generate the missing checkpoints before rerunning for all cases.")


if __name__ == "__main__":
    main()
