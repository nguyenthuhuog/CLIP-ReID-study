import tempfile
import unittest
from pathlib import Path

from PIL import Image

from create_comparison_grid import (
    CELL_HEIGHT,
    CELL_WIDTH,
    HEADER_HEIGHT,
    MODELS,
    OUTER_PADDING,
    RATIOS,
    ROW_LABEL_WIDTH,
    TABLE_GAP,
    create_comparison_grids,
)


class ComparisonGridTest(unittest.TestCase):
    def test_creates_one_comparison_table_per_original_image(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            comparison_dir = root / "comparisons"
            originals_dir = comparison_dir / "originals"
            originals_dir.mkdir(parents=True)

            for image_index in range(3):
                key = "person_{}".format(image_index)
                Image.new("RGB", (16, 32), (20, 30, image_index)).save(
                    originals_dir / (key + ".png")
                )
                for ratio in RATIOS:
                    condition = "occlusion_{:03d}".format(ratio)
                    for model_index, (_, model_dir) in enumerate(MODELS):
                        overlay_dir = (
                            comparison_dir / "overlays" / model_dir / condition
                        )
                        overlay_dir.mkdir(parents=True, exist_ok=True)
                        Image.new(
                            "RGB",
                            (16, 32),
                            (ratio, model_index, image_index),
                        ).save(
                            overlay_dir / (key + "_attention.png")
                        )

            output_dir = root / "per_image"
            output_paths, dimensions = create_comparison_grids(
                comparison_dir, output_dir
            )

            self.assertEqual(len(output_paths), 3)
            self.assertEqual(
                {path.name for path in output_paths},
                {
                    "person_0_comparison.png",
                    "person_1_comparison.png",
                    "person_2_comparison.png",
                },
            )
            expected_dimensions = (
                OUTER_PADDING * 2
                + ROW_LABEL_WIDTH
                + TABLE_GAP
                + 4 * CELL_WIDTH
                + 3 * TABLE_GAP,
                OUTER_PADDING * 2 + HEADER_HEIGHT + 5 * CELL_HEIGHT,
            )
            self.assertEqual(dimensions, expected_dimensions)
            for output_path in output_paths:
                with Image.open(output_path) as result:
                    self.assertEqual(result.size, dimensions)
                    self.assertEqual(result.mode, "RGB")

            with Image.open(output_dir / "person_0_comparison.png") as result:
                first_original_pixel = result.getpixel(
                    (
                        OUTER_PADDING + ROW_LABEL_WIDTH + TABLE_GAP + 20,
                        OUTER_PADDING + HEADER_HEIGHT + 20,
                    )
                )
                second_original_pixel = result.getpixel(
                    (
                        OUTER_PADDING + ROW_LABEL_WIDTH + TABLE_GAP + 20,
                        OUTER_PADDING + HEADER_HEIGHT + CELL_HEIGHT + 20,
                    )
                )
                self.assertEqual(first_original_pixel, second_original_pixel)


if __name__ == "__main__":
    unittest.main()
