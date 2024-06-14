import math
from typing import Dict, List, Optional, Tuple
from omegaconf import OmegaConf
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as pth_transforms
import torchvision.transforms.functional as F

from lavis.processors.base_processor import BaseProcessor


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Ops which can be used on depth
DEPTH_OPS = [
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Rotate",
    "Invert",
    "Identity",
]


def _apply_op(
    img: torch.Tensor,
    op_name: str,
    magnitude: float,
    interpolation: F.InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(magnitude), 0.0],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "ShearY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(magnitude)],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        # The tensor dtype must be torch.uint8
        # and values are expected to be in [0, 255]
        img = (img * 255).to(dtype=torch.uint8)
        img = F.posterize(img, int(magnitude))
        img = (img / 255.0).to(dtype=torch.float32)
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        # The tensor dtype must be torch.uint8
        # and values are expected to be in [0, 255]
        img = (img * 255).to(dtype=torch.uint8)
        img = F.equalize(img)
        img = (img / 255.0).to(dtype=torch.float32)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class RandAugment3d(torch.nn.Module):
    """
    Wrapper around torchvision RandAugment transform
    to support 4 channel input for RGBD data

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: F.InterpolationMode = F.InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(
        self, num_bins: int, image_size: List[int]
    ) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins),
                True,
            ),
            "TranslateY": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins),
                True,
            ),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Transformed image.
        """
        assert isinstance(img, torch.Tensor)

        C, H, W = img.shape
        images = [img[:3, ...]]  # RGB

        if C == 4:
            depth = img[3:4, ...]  # (1, H, W)
            images.append(depth)

        # Select ops
        # We sample an op and its metadata so that the same op
        # is applied to both RGB and D where relevant
        selected_ops = []
        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(self.num_magnitude_bins, (H, W))
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            selected_ops.append(op_name)

        # Apply on both images and depth
        images_out = []
        for im in images:
            # Only apply some ops for depth if
            # they are part of DEPTH_OPS
            run_on_depth = C == 1 and op_name in DEPTH_OPS
            if C == 3 or run_on_depth:
                fill = self.fill
                if isinstance(im, torch.Tensor):
                    if isinstance(fill, (int, float)):
                        fill = [float(fill)] * C
                    elif fill is not None:
                        fill = [float(f) for f in fill]

                for op_name in selected_ops:
                    magnitudes, signed = op_meta[op_name]
                    magnitude = (
                        float(magnitudes[self.magnitude].item())
                        if magnitudes.ndim > 0
                        else 0.0
                    )
                    if signed and torch.randint(2, (1,)):
                        magnitude *= -1.0

                    im = _apply_op(
                        im,
                        op_name,
                        magnitude,
                        interpolation=self.interpolation,
                        fill=fill,
                    )

            # Save modified image
            images_out.append(im)

        # Concat the img and depth back if present
        images_out = torch.cat(images_out, dim=0)

        return images_out

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_ops={num_ops}"
        s += ", magnitude={magnitude}"
        s += ", num_magnitude_bins={num_magnitude_bins}"
        s += ", interpolation={interpolation}"
        s += ", fill={fill}"
        s += ")"
        return s.format(**self.__dict__)


class ColorJitter3d(pth_transforms.ColorJitter):
    """
    Apply ColorJitter on an image of shape (4, H, W)
    """

    def __init__(self, brightness, contrast, saturation, hue):
        """
        Args:
            strength (float): A number used to quantify the strength of the
                              color distortion.
        """
        super().__init__(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, image: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise ValueError("Expected tensor input")
        C, H, W = image.shape
        if C != 4:
            err_msg = "This transform is for 4 channel RGBD input only; got %d" % C
            raise ValueError(err_msg)
        color_img = image[:3, ...]  # (3, H, W)
        depth_img = image[3:4, ...]  # (1, H, W)
        color_img_jitter = super().__call__(color_img)
        img = torch.cat([color_img_jitter, depth_img], dim=0)

        return img


class DropChannels(torch.nn.Module):
    """
    Drops Channels with predefined probability values.
    Pads the dropped channels with `pad_value`.
    Channels can be tied using `tie_channels`
    For example, for RGBD input, RGB can be tied by using `tie_channels=[0,1,2]`.
    In this case, channels [0,1,2] will be dropped all at once or not at all.
    Assumes input is of the form CxHxW or TxCxHxW
    """

    def __init__(
        self, channel_probs, fill_values, tie_channels=None, all_channel_drop=False
    ):
        """
        channel_probs: List of probabilities
        fill_values: List of values to fill the dropped channels with
        tie_channels: List of indices. Tie dropping of certain channels.
        all_channel_drop: Bool variable to prevent cases where all channels are dropped.
        """
        super().__init__()
        assert len(channel_probs) == len(
            fill_values
        ), f"Mismatch in length of channel_probs and fill_values: {len(channel_probs)} vs. {len(fill_values)}"

        assert len(channel_probs) in [
            3,
            4,
        ], f"channel_probs length is {len(channel_probs)}. Should be 3 or 4"

        channel_probs = np.array(channel_probs, dtype=np.float32)
        assert np.all(channel_probs >= 0)
        assert np.all(channel_probs <= 1)

        self.channel_probs = channel_probs
        self.fill_values = fill_values
        self.tie_channels = tie_channels
        self.all_channel_drop = all_channel_drop

        if tie_channels is not None:
            assert len(tie_channels) <= len(channel_probs)
            assert max(tie_channels) < len(channel_probs)
            assert min(tie_channels) >= 0
            tie_probs = [channel_probs[x] for x in tie_channels]
            assert len(set(tie_probs)) == 1, "All tie_channel probs must be equal"

    def __call__(self, x):
        assert isinstance(x, torch.Tensor)
        if x.ndim == 3:
            # CxHxW
            num_channels = x.shape[0]
            channel_index = 0
        elif x.ndim == 4:
            # TxCxHxW
            num_channels = x.shape[1]
            channel_index = 1
        else:
            raise ValueError(f"Unexpected number of dims {x.ndim}. Expected 3 or 4.")

        assert num_channels == len(
            self.channel_probs
        ), f"channel_probs is {len(self.channel_probs)} but got {num_channels} channels"

        to_drop = [
            np.random.random() < self.channel_probs[c] for c in range(num_channels)
        ]
        if self.tie_channels is not None:
            first_drop = to_drop[self.tie_channels[0]]
            for idx in self.tie_channels[1:]:
                to_drop[idx] = first_drop

        if all(to_drop) and self.all_channel_drop is False:
            # all channels will be dropped, prevent it
            to_drop = [False for _ in range(num_channels)]

        for c in range(num_channels):
            if not to_drop[c]:
                continue
            if channel_index == 0:
                x[c, ...] = self.fill_values[c]
            elif channel_index == 1:
                x[:, c, ...] = self.fill_values[c]
            else:
                raise NotImplementedError()
        return x


class DepthNorm(torch.nn.Module):
    """
    Normalize the depth channel: in an RGBD input of shape (4, H, W),
    only the last channel is modified.
    The depth channel is also clamped at 0.0. The Midas depth prediction
    model outputs inverse depth maps - negative values correspond
    to distances far away so can be clamped at 0.0
    """

    def __init__(
        self,
        max_depth: float,
        clamp_max_before_scale: bool = False,
        min_depth: float = 0.01,
    ):
        """
        Args:
            max_depth (float): The max value of depth for the dataset
            clamp_max (bool): Whether to clamp to max_depth or to divide by max_depth
        """
        super().__init__()
        if max_depth < 0.0:
            raise ValueError("max_depth must be > 0; got %.2f" % max_depth)
        self.max_depth = max_depth
        self.clamp_max_before_scale = clamp_max_before_scale
        self.min_depth = min_depth

    def __call__(self, image: torch.Tensor):
        C, H, W = image.shape
        if C != 4:
            err_msg = (
                f"This transform is for 4 channel RGBD input only; got {image.shape}"
            )
            raise ValueError(err_msg)
        color_img = image[:3, ...]  # (3, H, W)
        depth_img = image[3:4, ...]  # (1, H, W)

        # Clamp to 0.0 to prevent negative depth values
        depth_img = depth_img.clamp(min=self.min_depth)

        # divide by max_depth
        if self.clamp_max_before_scale:
            depth_img = depth_img.clamp(max=self.max_depth)

        depth_img /= self.max_depth

        img = torch.cat([color_img, depth_img], dim=0)
        return img


rgbd_conf = OmegaConf.create(
    {
        "max_depth": 75,
        "clamp_max_before_scale": True,
        "num_ops": 2,
        "magnitude": 9,
    }
)

class RGBD_Processor_Train(BaseProcessor):
    def __init__(self, args, img_mean=None, img_std=None, depth_mean=None, depth_std=None, update_conf=None):
        img_mean = img_mean if img_mean is not None else OPENAI_CLIP_MEAN
        img_std = img_std if img_std is not None else OPENAI_CLIP_STD
        depth_mean = depth_mean if depth_mean is not None else 0.0
        depth_std = depth_std if depth_std is not None else 1.0
        
        self.mean = list(img_mean) + [depth_mean]
        self.std = list(img_std) + [depth_std]
        
        if update_conf is not None:
            args.update(update_conf)

        self.img_to_tensor = pth_transforms.ToTensor()

        self.rgbd_transform = pth_transforms.Compose([
            DepthNorm(
                max_depth=args.max_depth,
                clamp_max_before_scale=args.clamp_max_before_scale,
            ),
            pth_transforms.RandomResizedCrop(
                size=224,
                interpolation=2
            ),
            pth_transforms.RandomHorizontalFlip(),
            RandAugment3d(
                num_ops=args.num_ops,
                magnitude=args.magnitude,
                interpolation=2,
            ),
            ColorJitter3d(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.4
            ),
            pth_transforms.RandomErasing(
                p=0.25
            ),
            pth_transforms.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])

    def __call__(self, img_path, depth_path):
        # here depth refers to disparity, in torch savefile format
        # note use ToTensor to scale image to [0,1] first
        img = Image.open(img_path).convert("RGB")
        img = self.img_to_tensor(img)
        
        depth = torch.load(depth_path)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)

        rgbd = torch.cat([img, depth], dim=0)
        transform_rgbd = self.rgbd_transform(rgbd)
        img = transform_rgbd[0:3, ...]
        depth = transform_rgbd[3:4, ...]
 
        return img, depth

    def __repr__(self):
        repr = "(DataAugmentationForRGBD,\n"
        repr += "transform = %s,\n" % str(self.rgbd_transform)
        repr += ")"
        return repr
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", rgbd_conf)
        img_mean = cfg.get("img_mean", OPENAI_CLIP_MEAN)
        img_std = cfg.get("img_std", OPENAI_CLIP_STD)
        depth_mean = cfg.get("depth_mean", 0.0418)
        depth_std = cfg.get("depth_std", 0.0295)

        return cls(args=args,img_mean=img_mean,img_std=img_std, depth_mean=depth_mean, depth_std=depth_std, update_conf=update_conf)
    
    
class RGBD_Processor_Eval(BaseProcessor):
    def __init__(self, args, img_mean=None, img_std=None, depth_mean=None, depth_std=None, update_conf=None):
        img_mean = img_mean if img_mean is not None else OPENAI_CLIP_MEAN
        img_std = img_std if img_std is not None else OPENAI_CLIP_STD
        depth_mean = depth_mean if depth_mean is not None else 0.0
        depth_std = depth_std if depth_std is not None else 1.0
        
        self.mean = list(img_mean) + [depth_mean]
        self.std = list(img_std) + [depth_std]
        
        if update_conf is not None:
            args.update(update_conf)

        self.img_to_tensor = pth_transforms.ToTensor()

        self.rgbd_transform = pth_transforms.Compose([
            DepthNorm(
                max_depth=args.max_depth,
                clamp_max_before_scale=args.clamp_max_before_scale,
            ),
            pth_transforms.Resize(
                size=224,
                interpolation=3
            ),
            pth_transforms.CenterCrop(
                size=224,
            ),
            pth_transforms.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])

    def __call__(self, img_path, depth_path):
        # here depth refers to disparity, in torch savefile format
        # note use ToTensor to scale image to [0,1] first
        img = Image.open(img_path).convert("RGB")
        img = self.img_to_tensor(img)
        
        depth = torch.load(depth_path)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)

        rgbd = torch.cat([img, depth], dim=0)
        transform_rgbd = self.rgbd_transform(rgbd)
        img = transform_rgbd[0:3, ...]
        depth = transform_rgbd[3:4, ...]
 
        return img, depth

    def __repr__(self):
        repr = "(DataAugmentationForRGBD,\n"
        repr += "transform = %s,\n" % str(self.rgbd_transform)
        repr += ")"
        return repr
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", rgbd_conf)
        img_mean = cfg.get("img_mean", OPENAI_CLIP_MEAN)
        img_std = cfg.get("img_std", OPENAI_CLIP_STD)
        depth_mean = cfg.get("depth_mean", 0.0418)
        depth_std = cfg.get("depth_std", 0.0295)

        return cls(args=args,img_mean=img_mean,img_std=img_std, depth_mean=depth_mean, depth_std=depth_std, update_conf=update_conf)