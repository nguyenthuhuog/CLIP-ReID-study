import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from create_attention_overlays import create_next_overlay_part
from generate_attention_results import create_original_overlays
from utils.attention_maps import get_attention_run_dir, save_attention_batch

try:
    import torch
    from model.clip.model import AttentionPool2d, VisionTransformer
except ImportError:
    torch = None


class AttentionMapTest(unittest.TestCase):
    def test_comparison_overlay_uses_patched_input_and_attention_map(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            condition_dir = root / "occlusion_020"
            original_dir = condition_dir / "originals"
            patched_dir = condition_dir / "patched"
            maps_dir = condition_dir / "maps"
            for directory in (original_dir, patched_dir, maps_dir):
                directory.mkdir(parents=True)

            original_path = original_dir / "image.png"
            patched_path = patched_dir / "image.png"
            attention_path = maps_dir / "image.png"
            Image.new("RGB", (8, 8), (0, 0, 255)).save(original_path)
            patched = Image.new("RGB", (8, 8), (0, 255, 0))
            for y in range(8):
                for x in range(4):
                    patched.putpixel((x, y), (0, 0, 0))
            patched.save(patched_path)
            Image.new("L", (2, 2), 255).save(attention_path)
            (condition_dir / "index.jsonl").write_text(
                json.dumps(
                    {
                        "key": "image",
                        "original": "originals/image.png",
                        "patched": "patched/image.png",
                        "attention": "maps/image.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            count, output_dir = create_original_overlays(
                condition_dir, root / "comparisons", "cnn_top_k_8", 0.5
            )

            self.assertEqual(count, 1)
            with Image.open(root / "comparisons" / "originals" / "image.png") as image:
                self.assertEqual(image.getpixel((1, 1)), (0, 0, 255))
            with Image.open(output_dir / "image_attention.png") as image:
                self.assertEqual(image.getpixel((1, 1)), (127, 20, 0))
                self.assertEqual(image.getpixel((6, 1)), (127, 147, 0))

    @unittest.skipIf(torch is None, "PyTorch is required for model attention tests.")
    def test_vit_returns_final_cls_to_patch_attention(self):
        vision = VisionTransformer(
            h_resolution=2,
            w_resolution=2,
            patch_size=16,
            stride_size=16,
            width=64,
            layers=12,
            heads=1,
            output_dim=16,
        ).eval()
        vision.set_attention_capture(True)

        with torch.no_grad():
            vision(torch.rand(1, 3, 32, 32))
        attention = vision.get_attention_map()

        self.assertEqual(tuple(attention.shape), (1, 2, 2))
        self.assertTrue(torch.isfinite(attention).all())

    @unittest.skipIf(torch is None, "PyTorch is required for model attention tests.")
    def test_resnet_attention_pool_returns_spatial_attention(self):
        pool = AttentionPool2d(
            spacial_dim=4, embed_dim=8, num_heads=2, output_dim=4
        ).eval()
        pool_input = torch.rand(2, 8, 2, 2)
        pool.capture_attention = True

        with torch.no_grad():
            pool(pool_input)
        attention = pool.get_attention_map()

        self.assertEqual(tuple(attention.shape), (2, 2, 2))
        self.assertTrue(torch.isfinite(attention).all())

    @unittest.skipIf(torch is None, "PyTorch is required for image conversion tests.")
    def test_saves_grayscale_map_and_patched_visualizations(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jpg"
            Image.new("RGB", (20, 30), (100, 120, 140)).save(source)
            cfg = SimpleNamespace(
                TEST=SimpleNamespace(
                    WEIGHT="checkpoint.pth",
                    ATTENTION=SimpleNamespace(
                        MAP_DIR=str(root / "attention"),
                        MAP_SIZE=[4, 2],
                        OVERLAY_ALPHA=0.5,
                    ),
                ),
                DATASETS=SimpleNamespace(NAMES="market1501"),
                MODEL=SimpleNamespace(NAME="ViT-B-16"),
                INPUT=SimpleNamespace(
                    PIXEL_MEAN=[0.5, 0.5, 0.5],
                    PIXEL_STD=[0.5, 0.5, 0.5],
                ),
            )

            save_attention_batch(
                cfg,
                [str(source)],
                torch.zeros(1, 3, 8, 4),
                torch.tensor([[[0.0, 0.5], [0.75, 1.0]]]),
                0.2,
            )

            condition_dir = get_attention_run_dir(cfg, 0.2)
            record = json.loads(
                (condition_dir / "index.jsonl").read_text(encoding="utf-8")
            )
            with Image.open(condition_dir / record["attention"]) as image:
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, (2, 4))
            self.assertTrue((condition_dir / record["original"]).is_file())
            self.assertTrue((condition_dir / record["patched"]).is_file())
            self.assertTrue(
                (condition_dir / "patched_with_attention" / (record["key"] + ".png")).is_file()
            )

    def test_overlay_batches_are_randomly_partitioned_without_repeats(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            condition_dirs = [root / "condition_0", root / "condition_20"]
            for condition_index, attention_dir in enumerate(condition_dirs):
                maps_dir = attention_dir / "maps"
                maps_dir.mkdir(parents=True)
                index_path = attention_dir / "index.jsonl"
                with index_path.open("w", encoding="utf-8") as index_file:
                    for index in range(10):
                        source = root / "image_{}.jpg".format(index)
                        if not source.exists():
                            Image.new(
                                "RGB", (12, 12), (index * 10, 20, 30)
                            ).save(source)
                        map_path = maps_dir / "map_{}.png".format(index)
                        Image.new(
                            "L", (2, 2), index * 20 + condition_index
                        ).save(map_path)
                        index_file.write(
                            json.dumps(
                                {
                                    "key": "image_{}".format(index),
                                    "source_path": str(source),
                                    "attention": "maps/{}".format(map_path.name),
                                }
                            )
                            + "\n"
                        )

            produced = set()
            for expected_part in range(1, 11):
                attention_dir = condition_dirs[expected_part % 2]
                output_dir = attention_dir / "original_with_attention"
                previous_outputs = (
                    set(output_dir.iterdir()) if output_dir.exists() else set()
                )
                part_number, image_count, _ = create_next_overlay_part(
                    attention_dir, output_dir, 10, 0.5
                )
                new_outputs = set(output_dir.iterdir()) - previous_outputs
                self.assertEqual(part_number, expected_part)
                self.assertEqual(image_count, 1)
                self.assertEqual(len(new_outputs), 1)
                new_output = next(iter(new_outputs))
                self.assertNotIn(new_output.stem, produced)
                produced.add(new_output.stem)

            self.assertEqual(len(produced), 10)
            with self.assertRaises(RuntimeError):
                create_next_overlay_part(
                    condition_dirs[0],
                    condition_dirs[0] / "original_with_attention",
                    10,
                    0.5,
                )


if __name__ == "__main__":
    unittest.main()
