import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RESULTS_PATH = ROOT / "results" / "cnn_stage2_epoch_metrics.csv"
RESULT_FIELDS = (
    "run_id",
    "experiment_name",
    "model",
    "method",
    "top_k",
    "epoch",
    "mAP_percent",
    "rank1_percent",
    "rank5_percent",
    "stage2_epoch_training_seconds",
    "stage2_cumulative_training_seconds",
    "stage2_peak_gpu_memory_mb",
    "checkpoint",
)

BASELINE_OPTS = (
    "I2T_TOPK.ENABLED",
    "False",
    "I2T_RANDOM_POOL.ENABLED",
    "False",
    "I2T_RANDOM_K.ENABLED",
    "False",
)

CNN_CONFIG = "configs/person/cnn_clipreid.yml"
TOP_K_VALUES = (8, 16)
EXPERIMENTS = (
    ("cnn", CNN_CONFIG, "baseline", None, BASELINE_OPTS),
) + tuple(
    (
        "cnn",
        CNN_CONFIG,
        "top_k",
        k,
        (
            "I2T_TOPK.ENABLED",
            "True",
            "I2T_RANDOM_POOL.ENABLED",
            "False",
            "I2T_RANDOM_K.ENABLED",
            "False",
            "I2T_TOPK.NUMBER_K",
            str(k),
        ),
    )
    for k in TOP_K_VALUES
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


def read_stage2_epoch_metrics(metrics_path):
    if not metrics_path.is_file():
        raise FileNotFoundError(
            "Stage 2 epoch metrics were not saved: {}".format(metrics_path)
        )

    required_fields = {
        "epoch",
        "mAP_percent",
        "rank1_percent",
        "rank5_percent",
        "stage2_epoch_training_seconds",
        "stage2_cumulative_training_seconds",
        "stage2_peak_gpu_memory_mb",
    }
    with metrics_path.open("r", newline="", encoding="utf-8") as metrics_file:
        reader = csv.DictReader(metrics_file)
        missing_fields = required_fields.difference(reader.fieldnames or ())
        if missing_fields:
            raise RuntimeError(
                "Missing fields {} in {}".format(
                    sorted(missing_fields), metrics_path
                )
            )
        rows = list(reader)

    if not rows:
        raise RuntimeError("No Stage 2 epoch metrics found in {}".format(metrics_path))
    epochs = [int(row["epoch"]) for row in rows]
    if epochs[0] != 1 or epochs != list(range(1, epochs[-1] + 1)):
        raise RuntimeError(
            "Stage 2 metrics have missing or unordered epochs in {}".format(
                metrics_path
            )
        )
    return rows


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

    for model_name, config_path, method_name, top_k, method_opts in EXPERIMENTS:
        case_name = "{}_{}".format(model_name, method_name)
        if top_k is not None:
            case_name += "_{}".format(top_k)
        case_output = output_root / case_name
        case_output.mkdir(parents=True, exist_ok=False)
        train_log = logs_root / (case_name + "_train.log")
        generated_checkpoint = case_output / "RN50_final.pth"
        checkpoint = case_output / (case_name + "_final.pth")

        print(
            "[INFO] Training {} ({}, K={})".format(
                model_name,
                method_name,
                top_k if top_k is not None else "all classes",
            )
        )
        train_command = [
            str(PYTHON),
            "train_clipreid.py",
            "--config_file",
            config_path,
            "OUTPUT_DIR",
            str(case_output),
            "SOLVER.STAGE2.EVAL_PERIOD",
            "1",
        ] + list(method_opts)
        run_logged(train_command, train_log, env)

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
        epoch_metrics = read_stage2_epoch_metrics(
            case_output / "stage2_epoch_metrics.csv"
        )
        rows = [
            {
                "run_id": run_id,
                "experiment_name": case_name,
                "model": model_name,
                "method": method_name,
                "top_k": top_k if top_k is not None else "",
                **result,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
            }
            for result in epoch_metrics
        ]
        append_results(rows)
        print(
            "[INFO] Appended {} Stage 2 epoch results for {} to {}".format(
                len(rows), case_name, RESULTS_PATH
            )
        )

    print(
        "[INFO] All {} CNN training cases completed sequentially.".format(
            len(EXPERIMENTS)
        )
    )
    print("[INFO] Results: {}".format(RESULTS_PATH))


if __name__ == "__main__":
    main()
