"""Common torchvision-style transform factories."""

from __future__ import annotations

from typing import Callable

try:
    from torchvision import transforms as T

    def get_transforms(
        *,
        image_size: int = 224,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
        augment: bool = False,
    ) -> dict[str, Callable]:
        """Return a dict of ``{"train": …, "val": …}`` transforms.

        Args:
            image_size: Target spatial resolution.
            mean:       Channel-wise normalisation mean.
            std:        Channel-wise normalisation std.
            augment:    Whether to add random augmentations for training.
        """
        normalise = T.Normalize(mean=mean, std=std)

        train_tf: list = [T.Resize((image_size, image_size))]
        if augment:
            train_tf += [
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        train_tf += [T.ToTensor(), normalise]

        val_tf = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalise,
        ]

        return {
            "train": T.Compose(train_tf),
            "val": T.Compose(val_tf),
        }

except ImportError:  # torchvision not available — return identity transforms

    def get_transforms(**kwargs) -> dict[str, Callable]:  # type: ignore[misc]
        """Fallback identity transforms (torchvision not installed)."""
        identity: Callable = lambda x: x  # noqa: E731
        return {"train": identity, "val": identity}
