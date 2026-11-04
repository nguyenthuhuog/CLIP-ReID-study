import unittest

from PIL import Image

from datasets.preprocessing import RandomPatchOcclusion


class RandomPatchOcclusionTest(unittest.TestCase):
    def test_occludes_requested_pixel_area_with_black(self):
        for size in ((37, 61), (128, 256)):
            image = Image.new("RGB", size, (200, 100, 50))

            for ratio in (0.2, 0.4, 0.6):
                with self.subTest(size=size, ratio=ratio):
                    for _ in range(5):
                        result = RandomPatchOcclusion(ratio)(image)
                        black_pixels = result.tobytes().count(b"\x00\x00\x00")
                        self.assertEqual(
                            black_pixels, round(ratio * size[0] * size[1])
                        )

    def test_zero_ratio_leaves_image_unchanged(self):
        image = Image.new("RGB", (20, 30), (200, 100, 50))

        self.assertEqual(
            RandomPatchOcclusion(0)(image).tobytes(),
            image.tobytes(),
        )

    def test_rejects_ratio_outside_zero_to_one(self):
        for ratio in (-0.1, 1.1):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    RandomPatchOcclusion(ratio)


if __name__ == "__main__":
    unittest.main()
