import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

from PIL import Image

from utils.attention_maps import colorize_attention_overlay


def _read_records(attention_dir):
    records = {}
    index_path = attention_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError("Attention index not found: {}".format(index_path))

    with index_path.open("r", encoding="utf-8") as index_file:
        for line in index_file:
            record = json.loads(line)
            source_path = Path(record["source_path"])
            map_path = attention_dir / record["attention"]
            if source_path.is_file() and map_path.is_file():
                records[record["key"]] = record

    if not records:
        raise RuntimeError("No valid attention maps were found in {}.".format(attention_dir))
    return records


def _partition(keys, count):
    shuffled = list(keys)
    random.SystemRandom().shuffle(shuffled)
    base_size, remainder = divmod(len(shuffled), count)
    partitions = []
    start = 0
    for index in range(count):
        size = base_size + (1 if index < remainder else 0)
        partitions.append(shuffled[start:start + size])
        start += size
    return partitions


def _fingerprint(keys):
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_state(state_path, state):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(state_path.parent), delete=False
    ) as temporary_file:
        json.dump(state, temporary_file, ensure_ascii=False, indent=2)
        temporary_path = temporary_file.name
    os.replace(temporary_path, state_path)


def create_next_overlay_part(attention_dir, output_dir, part_count, alpha):
    if part_count <= 0:
        raise ValueError("The number of parts must be positive.")
    if not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be between 0 and 1.")

    records = _read_records(attention_dir)
    keys = sorted(records)
    current_fingerprint = _fingerprint(keys)
    state_path = attention_dir.parent / ".overlay_progress.json"

    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    else:
        state = {}

    if state.get("fingerprint") != current_fingerprint or state.get("part_count") != part_count:
        completed = set(state.get("completed_keys", []))
        pending = [key for key in keys if key not in completed]
        state = {
            "fingerprint": current_fingerprint,
            "part_count": part_count,
            "next_part": 0,
            "completed_keys": sorted(completed),
            "parts": _partition(pending, part_count),
        }

    next_part = state["next_part"]
    if next_part >= part_count:
        raise RuntimeError(
            "All {} parts have already been processed; no images will be repeated.".format(
                part_count
            )
        )

    selected_keys = state["parts"][next_part]
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in selected_keys:
        record = records[key]
        source_path = Path(record["source_path"])
        map_path = attention_dir / record["attention"]
        output_path = output_dir / (key + "_attention.png")
        if output_path.exists():
            continue

        with Image.open(source_path) as source_file:
            original = source_file.convert("RGB")
        with Image.open(map_path) as map_file:
            attention = map_file.convert("L").resize(
                original.size, Image.Resampling.BILINEAR
            )
        colorize_attention_overlay(original, attention, alpha).save(
            output_path, format="PNG"
        )

    completed = set(state["completed_keys"])
    completed.update(selected_keys)
    state["completed_keys"] = sorted(completed)
    state["next_part"] = next_part + 1
    _write_state(state_path, state)
    return next_part + 1, len(selected_keys), output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create the next non-repeating batch of attention overlays."
    )
    parser.add_argument(
        "--config_file",
        default="configs/person/vit_clipreid.yml",
        help="Test configuration used to locate attention maps.",
    )
    parser.add_argument(
        "--attention_dir",
        default="",
        help="Specific occlusion-condition directory containing index.jsonl.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.0,
        help="Occlusion ratio used when testing (for example 0.2).",
    )
    parser.add_argument(
        "--weight",
        default="",
        help="Checkpoint path used during testing, if supplied on the command line.",
    )
    parser.add_argument("--parts", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    if args.attention_dir:
        attention_dir = Path(args.attention_dir)
        alpha = args.alpha if args.alpha is not None else 0.45
    else:
        from config import cfg
        from utils.attention_maps import get_attention_run_dir

        cfg.merge_from_file(args.config_file)
        if args.weight:
            cfg.TEST.WEIGHT = args.weight
        cfg.freeze()
        attention_dir = get_attention_run_dir(cfg, args.ratio)
        alpha = (
            args.alpha
            if args.alpha is not None
            else cfg.TEST.ATTENTION.OVERLAY_ALPHA
        )

    output_dir = attention_dir / "original_with_attention"
    part_number, image_count, output_dir = create_next_overlay_part(
        attention_dir, output_dir, args.parts, alpha
    )
    print(
        "Created part {}/{} ({} images) in {}".format(
            part_number, args.parts, image_count, output_dir
        )
    )


if __name__ == "__main__":
    main()
