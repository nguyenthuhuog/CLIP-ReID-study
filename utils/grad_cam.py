import torch
from torch import nn


class _ReIDLogitsModel(nn.Module):
    def __init__(self, model, cam_label=None, view_label=None):
        super().__init__()
        self.model = model
        self.cam_label = cam_label
        self.view_label = view_label

    def forward(self, images):
        batch_size = images.shape[0]
        cam_label = (
            self.cam_label[:batch_size] if self.cam_label is not None else None
        )
        view_label = (
            self.view_label[:batch_size] if self.view_label is not None else None
        )
        return self.model(
            images,
            cam_label=cam_label,
            view_label=view_label,
            get_cam_logits=True,
        )


def create_reid_grad_cam(model, cam_label=None, view_label=None):
    from pytorch_grad_cam import GradCAM

    reid_model = model.module if isinstance(model, nn.DataParallel) else model
    image_encoder = reid_model.image_encoder
    cam_model = _ReIDLogitsModel(model, cam_label, view_label)

    if reid_model.model_name == "RN50":
        target_layers = [image_encoder.layer4[-1]]
        reshape_transform = None
    elif reid_model.model_name == "ViT-B-16":
        height = image_encoder.h_resolution
        width = image_encoder.w_resolution
        target_layers = [image_encoder.transformer.resblocks[-1].ln_1]

        def reshape_transform(activation):
            activation = activation.permute(1, 0, 2)[:, 1:, :]
            activation = activation.reshape(
                activation.shape[0], height, width, activation.shape[2]
            )
            return activation.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            "Grad-CAM does not support model backbone {}.".format(
                reid_model.model_name
            )
        )

    return GradCAM(
        model=cam_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )


def compute_reid_grad_cam(cam, images):
    grayscale_cams = cam(input_tensor=images, targets=None)
    return torch.from_numpy(grayscale_cams)
