import os
import random

import numpy as np
import torch


CHECKPOINT_VERSION = 2


def get_model_state_dict(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.state_dict()


def extract_model_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dictionary.")
    if "format_version" in checkpoint:
        if checkpoint["format_version"] != CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported checkpoint format version: {}".format(
                    checkpoint["format_version"]
                )
            )
        state_dict = checkpoint.get("model")
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain model weights.")
    return state_dict


def load_stage2_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Stage 2 resume checkpoint must be a dictionary.")
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Cannot resume from a legacy or unsupported checkpoint. "
            "Stage 2 resume requires checkpoint format version {}.".format(
                CHECKPOINT_VERSION
            )
        )
    if checkpoint.get("stage") != "stage2":
        raise ValueError("Resume checkpoint was not saved during Stage 2.")
    required_keys = {
        "epoch",
        "model",
        "center_criterion",
        "optimizer",
        "optimizer_center",
        "scheduler",
        "scaler",
        "rng_state",
        "stage2_metrics",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            "Stage 2 checkpoint is missing required state: {}".format(
                ", ".join(sorted(missing_keys))
            )
        )
    return checkpoint


def make_stage2_checkpoint(
    epoch,
    model,
    center_criterion,
    optimizer,
    optimizer_center,
    scheduler,
    scaler,
    stage2_metrics=None,
):
    numpy_rng_state = np.random.get_state()
    return {
        "format_version": CHECKPOINT_VERSION,
        "stage": "stage2",
        "epoch": epoch,
        "model": get_model_state_dict(model),
        "center_criterion": center_criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_center": optimizer_center.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": {
                "algorithm": numpy_rng_state[0],
                "state": numpy_rng_state[1].tolist(),
                "position": numpy_rng_state[2],
                "has_gauss": numpy_rng_state[3],
                "cached_gaussian": numpy_rng_state[4],
            },
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "stage2_metrics": stage2_metrics or {
            "cumulative_training_seconds": 0.0,
            "peak_gpu_memory_mb": 0.0,
        },
    }


def save_checkpoint(checkpoint, path):
    temporary_path = "{}.tmp".format(path)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def restore_rng_state(rng_state):
    random.setstate(rng_state["python"])
    numpy_state = rng_state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(rng_state["torch"])
    cuda_states = rng_state["cuda"]
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Cannot restore CUDA RNG state: checkpoint has {} devices, "
                "but {} are available.".format(
                    len(cuda_states), torch.cuda.device_count()
                )
            )
        torch.cuda.set_rng_state_all(cuda_states)
