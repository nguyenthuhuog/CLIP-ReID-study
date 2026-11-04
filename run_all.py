import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RESULTS_PATH = ROOT / "results" / "benchmark_results.csv"
OCCLUSION_RATIOS = (0.2, 0.4, 0.6, 0.8)
RESULT_FIELDS = (
    "run_id",
    "experiment_name",
    "model",
    "method",
    "mode",
    "occlusion_ratio_percent",
    "mAP_percent",
    "rank1_percent",
    "rank5_percent",
    "total_training_time_seconds",
    "peak_gpu_memory_mb",
    "checkpoint",
)

METHODS = (
    (
        "random_k",
        (
            "I2T_TOPK.ENABLED",
            "False",
            "I2T_RANDOM_POOL.ENABLED",
            "False",
            "I2T_RANDOM_K.ENABLED",
            "True",
        ),
    ),
    (
        "top_k",
        (
            "I2T_TOPK.ENABLED",
            "True",
            "I2T_RANDOM_POOL.ENABLED",
            "False",
            "I2T_RANDOM_K.ENABLED",
            "False",
        ),
    ),
    (
        "top_k_random_pool",
        (
            "I2T_TOPK.ENABLED",
            "True",
            "I2T_RANDOM_POOL.ENABLED",
            "True",
            "I2T_RANDOM_K.ENABLED",
            "False",
        ),
    ),
)

MODELS = (
    ("cnn", "configs/person/cnn_clipreid.yml"),
    ("vit", "configs/person/vit_clipreid.yml"),
)


def run_logged(command, log_path, env):
    print("[RUN]", " ".join(str(part) for part in command))
    with log_path.open("w", encoding="utf-8") as log_file:
        start_time = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.perf_counter() - start_time

    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            "Command failed with exit code {}. Log: {}\n{}".format(
                completed.returncode, log_path, "\n".join(tail)
            )
        )
    return elapsed


def read_peak_gpu_memory_mb(log_path):
    pattern = re.compile(
        r"peak GPU memory allocated:\s*([0-9]+(?:\.[0-9]+)?)\s*MB",
        re.IGNORECASE,
    )
    values = [
        float(match.group(1))
        for match in pattern.finditer(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
    ]
    if not values:
        raise RuntimeError("Training log has no peak GPU memory measurement: {}".format(log_path))
    return max(values)


def read_test_metrics(log_path):
    condition_pattern = re.compile(
        r"Evaluation condition:\s*([^;]+);\s*occlusion_ratio=([0-9.]+)"
    )
    metric_patterns = {
        "mAP_percent": re.compile(r"\bmAP:\s*([0-9]+(?:\.[0-9]+)?)%"),
        "rank1_percent": re.compile(
            r"CMC curve,\s*Rank-1\s*:\s*([0-9]+(?:\.[0-9]+)?)%"
        ),
        "rank5_percent": re.compile(
            r"CMC curve,\s*Rank-5\s*:\s*([0-9]+(?:\.[0-9]+)?)%"
        ),
    }

    results = []
    current = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        condition_match = condition_pattern.search(line)
        if condition_match:
            if current is not None:
                results.append(current)
            current = {
                "mode": condition_match.group(1).strip(),
                "occlusion_ratio_percent": float(condition_match.group(2)) * 100,
            }
            continue

        if current is not None:
            for name, pattern in metric_patterns.items():
                match = pattern.search(line)
                if match:
                    current[name] = float(match.group(1))

    if current is not None:
        results.append(current)

    required_metrics = set(metric_patterns)
    for result in results:
        missing = required_metrics.difference(result)
        if missing:
            raise RuntimeError(
                "Missing test metrics {} in {}".format(sorted(missing), log_path)
            )
    expected_count = len(OCCLUSION_RATIOS) + 1
    if len(results) != expected_count:
        raise RuntimeError(
            "Expected {} test results in {}, found {}".format(
                expected_count, log_path, len(results)
            )
        )
    return results


def append_results(rows):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_PATH.exists()
    if file_exists and RESULTS_PATH.stat().st_size:
        with RESULTS_PATH.open("r", newline="", encoding="utf-8") as result_file:
            existing_fields = next(csv.reader(result_file), [])
        if existing_fields != list(RESULT_FIELDS):
            raise RuntimeError(
                "Existing results file has a different schema: {}".format(
                    RESULTS_PATH
                )
            )

    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        if not file_exists or RESULTS_PATH.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)
        result_file.flush()


def main():
    if not PYTHON.is_file():
        raise FileNotFoundError(
            "Project virtual-environment Python was not found: {}".format(PYTHON)
        )

    run_id = "{}_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S"), os.getpid()
    )
    output_root = ROOT / "outputs" / run_id
    logs_root = ROOT / "logs" / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    logs_root.mkdir(parents=True, exist_ok=False)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    for model_name, config_path in MODELS:
        for method_name, method_opts in METHODS:
            case_name = "{}_{}".format(model_name, method_name)
            case_output = output_root / case_name
            case_output.mkdir(parents=True, exist_ok=False)
            train_log = logs_root / (case_name + "_train.log")
            test_log = logs_root / (case_name + "_test.log")
            model_checkpoint_name = (
                "RN50_final.pth"
                if model_name == "cnn"
                else "ViT-B-16_final.pth"
            )
            generated_checkpoint = case_output / model_checkpoint_name
            checkpoint = case_output / (case_name + "_final.pth")

            print("[INFO] Training {} ({})".format(model_name, method_name))
            train_command = [
                str(PYTHON),
                "train_clipreid.py",
                "--config_file",
                config_path,
                "OUTPUT_DIR",
                str(case_output),
            ] + list(method_opts)
            training_time = run_logged(train_command, train_log, env)

            if not generated_checkpoint.is_file():
                raise RuntimeError(
                    "Training completed without the expected checkpoint: {}".format(
                        generated_checkpoint
                    )
                )
            if checkpoint.exists():
                raise FileExistsError(
                    "Refusing to overwrite experiment checkpoint: {}".format(
                        checkpoint
                    )
                )
            generated_checkpoint.rename(checkpoint)
            peak_memory_mb = read_peak_gpu_memory_mb(
                case_output / "train_log.txt"
            )

            print("[INFO] Testing {} ({})".format(model_name, method_name))
            test_command = [
                str(PYTHON),
                "test_clipreid.py",
                "--config_file",
                config_path,
                "OUTPUT_DIR",
                str(case_output),
                "TEST.WEIGHT",
                str(checkpoint),
                "TEST.ATTENTION.ENABLED",
                "False",
                "TEST.OCCLUSION.ENABLED",
                "True",
                "TEST.OCCLUSION.RATIOS",
                str(list(OCCLUSION_RATIOS)),
            ]
            run_logged(test_command, test_log, env)

            test_results = read_test_metrics(case_output / "test_log.txt")
            rows = []
            for result in test_results:
                rows.append(
                    {
                        "run_id": run_id,
                        "experiment_name": case_name,
                        "model": model_name,
                        "method": method_name,
                        **result,
                        "total_training_time_seconds": round(training_time, 3),
                        "peak_gpu_memory_mb": round(peak_memory_mb, 2),
                        "checkpoint": str(checkpoint.relative_to(ROOT)),
                    }
                )
            append_results(rows)
            print(
                "[INFO] Appended {} test results for {} to {}".format(
                    len(rows), case_name, RESULTS_PATH
                )
            )

    print("[INFO] All six training/testing cases completed sequentially.")
    print("[INFO] Results: {}".format(RESULTS_PATH))


if __name__ == "__main__":
    main()
