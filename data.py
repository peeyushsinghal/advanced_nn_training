import torch
import yaml
import albumentations as A
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


class CIFAR10Data:

    def __init__(self) -> None:

        with open("config.yml", "r") as f:
            self.config = yaml.safe_load(f)

        torch.manual_seed(self.config["manual_seed"])

        # Set hardware settings based on device
        self.kwargs = (
            {
                "num_workers": self.config["num_workers"],
                "pin_memory": self.config["pin_memory"],
            }
            if self._get_device() == "cuda"
            else {}
        )

        self.train_transform, self.test_transform = self.apply_transforms()

    def apply_transforms(self):
        train_transform = A.Compose(
            [
                A.HorizontalFlip(p=self.config["p_horizonal_flip"]),
                A.ShiftScaleRotate(
                    shift_limit=[
                        self.config["shift_limit_left"],
                        self.config["shift_limit_right"],
                    ],
                    scale_limit=[
                        self.config["scale_limit_min"],
                        self.config["scale_limit_max"],
                    ],
                    rotate_limit=[
                        -1 * self.config["rotation_degrees"],
                        self.config["rotation_degrees"],
                    ],
                    p=self.config["p"],
                ),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=16,
                    max_width=16,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill=tuple((x * 255.0 for x in self.config["normalize_mean"])),
                    p=0.2,
                ),
                A.ToGray(p=0.15),
                A.Normalize(
                    mean=self.config["normalize_mean"],
                    std=self.config["normalize_std"],
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

        test_transform = A.Compose(
            [
                A.Normalize(
                    mean=self.config["normalize_mean"],
                    std=self.config["normalize_std"],
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

        # return train_transform, test_transform
        return (
            lambda img: train_transform(image=np.array(img))["image"],
            lambda img: test_transform(image=np.array(img))["image"],
        )

    def _get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_train_loader(self):

        train_dataset = CIFAR10(
            self.config["train_data_path"],
            train=True,
            download=True,
            transform=self.train_transform,
        )

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            **self.kwargs
        )

    def get_test_loader(self):
        test_dataset = CIFAR10(
            self.config["test_data_path"],
            train=False,
            download=True,
            transform=self.test_transform,
        )

        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            **self.kwargs
        )
