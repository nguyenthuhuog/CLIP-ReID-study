import unittest
from types import SimpleNamespace

import torch
from torch import nn

from utils.grad_cam import compute_reid_grad_cam, create_reid_grad_cam


class _TinyCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, images):
        return self.layer4(images)


class _TinyTransformerBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.proj = nn.Linear(width, width)

    def forward(self, tokens):
        return tokens + self.proj(self.ln_1(tokens))


class _TinyViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_resolution = 2
        self.w_resolution = 2
        self.patch_embed = nn.Conv2d(3, 4, kernel_size=4, stride=4)
        self.transformer = SimpleNamespace(
            resblocks=nn.ModuleList([_TinyTransformerBlock(4)])
        )

    def forward(self, images):
        patches = self.patch_embed(images).flatten(2).permute(2, 0, 1)
        cls_token = patches.mean(dim=0, keepdim=True)
        tokens = torch.cat((cls_token, patches), dim=0)
        tokens = self.transformer.resblocks[-1](tokens)
        return (
            tokens[1:]
            .permute(1, 2, 0)
            .reshape(images.shape[0], 4, self.h_resolution, self.w_resolution)
        )


class _TinyReIDModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.image_encoder = (
            _TinyCNNEncoder() if model_name == "RN50" else _TinyViTEncoder()
        )
        self.classifier = nn.Linear(4, 3)

    def forward(
        self,
        images,
        cam_label=None,
        view_label=None,
        get_cam_logits=False,
    ):
        features = self.image_encoder(images).mean(dim=(2, 3))
        return self.classifier(features)


class GradCAMTest(unittest.TestCase):
    def test_grad_cam_produces_spatial_maps_for_cnn_and_vit(self):
        for model_name in ("RN50", "ViT-B-16"):
            with self.subTest(model=model_name):
                model = _TinyReIDModel(model_name).eval()
                cam = create_reid_grad_cam(model)
                try:
                    with torch.enable_grad():
                        maps = compute_reid_grad_cam(
                            cam, torch.rand(2, 3, 8, 8)
                        )
                finally:
                    cam.activations_and_grads.release()

                self.assertEqual(tuple(maps.shape), (2, 8, 8))
                self.assertTrue(torch.isfinite(maps).all())
                self.assertTrue(torch.all(maps >= 0))
                self.assertTrue(torch.all(maps <= 1))


if __name__ == "__main__":
    unittest.main()
