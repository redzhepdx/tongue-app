import pydoc
import re
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn


def create_classification_model(h_params: Dict[str, Any], weight_path: str) -> nn.Module:
    model = timm.create_model(model_name=h_params["model_name"],
                              num_classes=h_params["classes"],
                              pretrained=True)
    # Load the pre-trained weights
    corrections: Dict[str, str] = {"model.": ""}
    state_dict = state_dict_from_disk(file_path=weight_path, rename_in_layers=corrections)
    model.load_state_dict(state_dict)

    model = nn.Sequential(model, nn.Softmax(dim=-1))

    model.to(torch.device("cuda:0"))

    return model


def create_segmentation_model(h_params: Dict[str, Any], weight_path: str) -> nn.Module:
    # Create a model
    model = object_from_dict(h_params)

    # Load the pre-trained weights
    corrections: Dict[str, str] = {"model.": ""}
    state_dict = state_dict_from_disk(file_path=weight_path, rename_in_layers=corrections)
    model.load_state_dict(state_dict)
    model.to(torch.device("cuda:0"))

    return model


def haze_removal(image, w0=0.6, t0=0.1):
    darkImage = image.min(axis=2)
    maxDarkChannel = darkImage.max()
    darkImage = darkImage.astype(np.double)

    t = 1 - w0 * (darkImage / maxDarkChannel)
    T = t * 255
    T.dtype = 'uint8'

    t[t < t0] = t0

    J = image
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * maxDarkChannel) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * maxDarkChannel) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * maxDarkChannel) / t

    return J


def fill_and_close(mask: np.ndarray) -> np.ndarray:
    # Thicken the edges
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=6)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.full_like(mask, 255)

    contours = list(filter(lambda c: cv2.contourArea(c) > 0, contours))
    contours = list(sorted(contours, key=lambda c: -1 * cv2.contourArea(c)))

    hull = cv2.convexHull(contours[0], False)

    if cv2.contourArea(hull) < (mask.shape[0] * mask.shape[1]) * 0.3:
        return np.full_like(mask, 255)

    cv2.drawContours(mask, contours[1:], 0, color=0, thickness=-1)
    cv2.drawContours(mask, [hull], 0, color=255, thickness=-1)

    return mask


def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def state_dict_from_disk(
        file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)

    return pydoc.locate(object_type)(**kwargs)


def denormalize(image: np.ndarray, mean: List[float] = [0.485, 0.456, 0.406],
                std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    return np.divide(np.subtract(image, mean), std)
