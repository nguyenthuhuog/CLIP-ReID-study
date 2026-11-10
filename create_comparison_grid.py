import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_COMPARISON_DIR = (
    "results/test_single/"
    "vit_20261110_095750_8524_20261110_175132/comparisons"
)
MODELS = (
    ("Baseline", "vit_baseline"),
    ("Top-K-8", "vit_top_k_8"),
    ("Top-K-16", "vit_top_k_16"),
)
RATIOS = (0, 10, 20, 30, 50)
CELL_WIDTH = 192
CELL_HEIGHT = 384
ROW_LABEL_WIDTH = 88
TABLE_GAP = 10
HEADER_TITLE_HEIGHT = 48
HEADER_HEIGHT = 112
OUTER_PADDING = 12


def _load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _rotated_text(text, font, fill):
    bounds = font.getbbox(text)
    text_image = Image.new(
        "RGBA",
        (bounds[2] - bounds[0], bounds[3] - bounds[1]),
        (0, 0, 0, 0),
    )
    ImageDraw.Draw(text_image).text(
        (-bounds[0], -bounds[1]), text, fill=fill, font=font
    )
    return text_image.rotate(90, expand=True)


def _validate_image_sets(comparison_dir):
    originals_dir = comparison_dir / "originals"
    original_paths = sorted(originals_dir.glob("*.png"))
    if not original_paths:
        raise FileNotFoundError(
            "No original PNG images found in {}".format(originals_dir)
        )

    image_keys = [path.stem for path in original_paths]
    sources = {}
    for ratio in RATIOS:
        condition = "occlusion_{:03d}".format(ratio)
        for label, model_dir in MODELS:
            overlay_dir = (
                comparison_dir / "overlays" / model_dir / condition
            )
            for key in image_keys:
                overlay_path = overlay_dir / (key + "_attention.png")
                if not overlay_path.is_file():
                    raise FileNotFoundError(
                        "Missing {} {} overlay: {}".format(
                            model_dir, condition, overlay_path
                        )
                    )
                sources[(key, label, ratio)] = overlay_path
    return image_keys, sources


def create_comparison_grids(comparison_dir, output_dir):
    comparison_dir = Path(comparison_dir)
    output_dir = Path(output_dir)
    image_keys, sources = _validate_image_sets(comparison_dir)
    headers = ("Original",) + tuple(label for label, _ in MODELS)

    table_width = (
        OUTER_PADDING * 2
        + ROW_LABEL_WIDTH
        + TABLE_GAP
        + len(headers) * CELL_WIDTH
        + (len(headers) - 1) * TABLE_GAP
    )
    table_height = (
        OUTER_PADDING * 2
        + HEADER_HEIGHT
        + len(RATIOS) * CELL_HEIGHT
    )
    header_font = _load_font(40)
    row_font = _load_font(36)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for key in image_keys:
        table = Image.new("RGB", (table_width, table_height), (238, 241, 245))
        draw = ImageDraw.Draw(table)
        original_path = comparison_dir / "originals" / (key + ".png")
        with Image.open(original_path) as image_file:
            original = image_file.convert("RGB").resize(
                (CELL_WIDTH, CELL_HEIGHT), Image.Resampling.LANCZOS
            )

        for column, header in enumerate(headers):
            x = (
                OUTER_PADDING
                + ROW_LABEL_WIDTH
                + TABLE_GAP
                + column * (CELL_WIDTH + TABLE_GAP)
            )
            draw.text(
                (
                    x + CELL_WIDTH // 2,
                    OUTER_PADDING
                    + HEADER_TITLE_HEIGHT
                    + (HEADER_HEIGHT - HEADER_TITLE_HEIGHT) // 2,
                ),
                header,
                fill=(30, 41, 59),
                font=header_font,
                anchor="mm",
            )

        # draw.text(
        #     (
        #         table_width // 2,
        #         OUTER_PADDING + HEADER_TITLE_HEIGHT // 2,
        #     ),
        #     "Image Occlusion Ratio (%)",
        #     fill=(30, 41, 59),
        #     font=header_font,
        #     anchor="mm",
        # )

        for row, ratio in enumerate(RATIOS):
            y = OUTER_PADDING + HEADER_HEIGHT + row * CELL_HEIGHT
            ratio_label = _rotated_text(
                "{}%".format(ratio), row_font, (30, 41, 59)
            )
            table.paste(
                ratio_label,
                (
                    OUTER_PADDING
                    + (ROW_LABEL_WIDTH - ratio_label.width) // 2,
                    y + (CELL_HEIGHT - ratio_label.height) // 2,
                ),
                ratio_label,
            )
            table.paste(
                original,
                (OUTER_PADDING + ROW_LABEL_WIDTH + TABLE_GAP, y),
            )
            for column, (label, _) in enumerate(MODELS, start=1):
                x = (
                    OUTER_PADDING
                    + ROW_LABEL_WIDTH
                    + TABLE_GAP
                    + column * (CELL_WIDTH + TABLE_GAP)
                )
                overlay_path = sources[(key, label, ratio)]
                with Image.open(overlay_path) as image_file:
                    overlay = image_file.convert("RGB").resize(
                        (CELL_WIDTH, CELL_HEIGHT), Image.Resampling.LANCZOS
                    )
                table.paste(overlay, (x, y))

        output_path = output_dir / (key + "_comparison.png")
        table.save(output_path, format="PNG")
        output_paths.append(output_path)

    return output_paths, table.size


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create one 5-row by 4-column comparison table per original image."
        )
    )
    parser.add_argument(
        "--comparison-dir",
        default=DEFAULT_COMPARISON_DIR,
        help="Folder containing originals/ and overlays/.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Folder for the per-image comparison tables (default: per_image).",
    )
    args = parser.parse_args()

    comparison_dir = Path(args.comparison_dir)
    if not comparison_dir.is_absolute():
        comparison_dir = ROOT / comparison_dir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else comparison_dir / "per_image"
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    output_paths, dimensions = create_comparison_grids(
        comparison_dir, output_dir
    )
    print(
        "[DONE] Created {} per-image comparison tables ({}x{}) in {}".format(
            len(output_paths), dimensions[0], dimensions[1], output_dir
        )
    )


if __name__ == "__main__":
    main()
