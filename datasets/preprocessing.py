import random
import math
from PIL import Image


class RandomPatchOcclusion:
    def __init__(self, ratio):
        if not 0 <= ratio <= 1:
            raise ValueError("Occlusion ratio must be between 0 and 1.")
        self.ratio = ratio

    def __call__(self, image):
        width, height = image.size
        target_pixels = round(self.ratio * width * height)
        if target_pixels == 0:
            return image

        occupied = [bytearray(width) for _ in range(height)]
        remaining = target_pixels
        patch_count = random.randint(1, min(4, remaining))

        for patch_index in range(patch_count):
            budget = max(1, remaining // (patch_count - patch_index))
            max_height = min(height, max(1, math.isqrt(budget)))
            placed = False

            for _ in range(50):
                patch_height = random.randint(1, max_height)
                patch_width = random.randint(1, min(width, budget // patch_height))
                top = random.randint(0, height - patch_height)
                left = random.randint(0, width - patch_width)
                if any(
                    any(row[left:left + patch_width])
                    for row in occupied[top:top + patch_height]
                ):
                    continue

                for row in occupied[top:top + patch_height]:
                    row[left:left + patch_width] = b"\xff" * patch_width
                remaining -= patch_height * patch_width
                placed = True
                break

            if not placed:
                break

        rows = list(range(height))
        random.shuffle(rows)
        for y in rows:
            if remaining == 0:
                break

            x = 0
            while x < width and remaining:
                while x < width and occupied[y][x]:
                    x += 1
                start = x
                while x < width and not occupied[y][x]:
                    x += 1
                available = x - start
                if available:
                    fill_width = min(available, remaining)
                    offset = random.randint(0, available - fill_width)
                    occupied[y][start + offset:start + offset + fill_width] = (
                        b"\xff" * fill_width
                    )
                    remaining -= fill_width

        mask = Image.new("L", image.size)
        mask.putdata([pixel for row in occupied for pixel in row])
        result = image.convert("RGB")
        result.paste((0, 0, 0), mask=mask)
        return result


def build_test_transform(size, pixel_mean, pixel_std, occlusion_ratio=None):
    import torchvision.transforms as T

    transforms = [T.Resize(size)]
    if occlusion_ratio is not None:
        transforms.append(RandomPatchOcclusion(occlusion_ratio))
    transforms.extend(
        [T.ToTensor(), T.Normalize(mean=pixel_mean, std=pixel_std)]
    )
    return T.Compose(transforms)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
