import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps


def _safe_component(value):
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return component or "unknown"


def attention_condition_name(ratio):
    return "occlusion_{:03d}".format(round(float(ratio) * 100))


def get_attention_run_dir(cfg, ratio):
    weight = Path(cfg.TEST.WEIGHT).stem if cfg.TEST.WEIGHT else "uncheckpointed"
    return (
        Path(cfg.TEST.ATTENTION.MAP_DIR)
        / _safe_component(cfg.DATASETS.NAMES)
        / _safe_component(cfg.MODEL.NAME)
        / _safe_component(weight)
        / attention_condition_name(ratio)
    )


def _image_key(source_path):
    normalized_path = os.path.normcase(os.path.abspath(source_path))
    digest = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:12]
    return "{}_{}".format(Path(source_path).stem, digest)


def _tensor_to_image(tensor, mean, std):
    tensor = tensor.detach().cpu().float()
    mean = tensor.new_tensor(mean).view(-1, 1, 1)
    std = tensor.new_tensor(std).view(-1, 1, 1)
    pixels = (tensor * std + mean).clamp(0, 1)
    array = pixels.mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def colorize_attention_overlay(image, attention, alpha):
    attention = attention.resize(image.size, Image.Resampling.BILINEAR)
    heatmap = ImageOps.colorize(
        attention, black=(0, 0, 0), white=(255, 40, 0)
    )
    return Image.blend(image.convert("RGB"), heatmap, alpha)


def save_attention_batch(cfg, image_paths, input_images, attention_maps, ratio):
    import numpy as np
    import torch.nn.functional as F

    map_size = tuple(int(value) for value in cfg.TEST.ATTENTION.MAP_SIZE)
    if len(map_size) != 2 or min(map_size) <= 0:
        raise ValueError("TEST.ATTENTION.MAP_SIZE must contain two positive values.")

    alpha = float(cfg.TEST.ATTENTION.OVERLAY_ALPHA)
    if not 0 <= alpha <= 1:
        raise ValueError("TEST.ATTENTION.OVERLAY_ALPHA must be between 0 and 1.")

    output_dir = get_attention_run_dir(cfg, ratio)
    attention_dir = output_dir / "maps"
    original_dir = output_dir / "originals"
    input_dir = output_dir / "inputs"
    patch_dir = output_dir / "patched"
    overlay_dir = output_dir / "patched_with_attention"
    for directory in (attention_dir, original_dir, input_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if ratio > 0:
        patch_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    maps = F.interpolate(
        attention_maps.detach().float().unsqueeze(1),
        size=map_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).cpu()
    records = []

    for index, source_path in enumerate(image_paths):
        key = _image_key(source_path)
        original_path = original_dir / (key + Path(source_path).suffix.lower())
        if not original_path.exists():
            shutil.copy2(source_path, original_path)

        model_input = _tensor_to_image(
            input_images[index], cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD
        )
        input_path = input_dir / (key + ".png")
        model_input.save(input_path, format="PNG")

        attention = maps[index].clamp(0, 1)
        attention_image = Image.fromarray(
            (attention.numpy() * 255).round().astype(np.uint8), mode="L"
        )
        map_path = attention_dir / (key + ".png")
        attention_image.save(map_path, format="PNG")

        patch_path = None
        if ratio > 0:
            patch_path = patch_dir / (key + ".png")
            model_input.save(patch_path, format="PNG")
            colorize_attention_overlay(
                model_input, attention_image, alpha
            ).save(overlay_dir / (key + ".png"), format="PNG")

        records.append(
            {
                "key": key,
                "source_path": os.path.abspath(source_path),
                "ratio": float(ratio),
                "attention": map_path.relative_to(output_dir).as_posix(),
                "original": original_path.relative_to(output_dir).as_posix(),
                "input": input_path.relative_to(output_dir).as_posix(),
                "patched": (
                    patch_path.relative_to(output_dir).as_posix()
                    if patch_path is not None
                    else None
                ),
            }
        )

    with (output_dir / "index.jsonl").open("a", encoding="utf-8") as index_file:
        for record in records:
            index_file.write(json.dumps(record, ensure_ascii=False) + "\n")
