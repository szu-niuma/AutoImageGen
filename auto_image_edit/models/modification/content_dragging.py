import random
from typing import List

import numpy as np
from pydantic import BaseModel, Field

from .base import ModificationBase


class RepContentDragModel(ModificationBase):
    name: str = "Content_Dragging"
    description: str = (
        "Procedure for dragging content within an image to a new location, replacing specific parts of an object, and subsequently modifying the object's attributes."
    )
    end_point: List[float] | None = Field(description="The end position of the dragging")

    @classmethod
    def find_random_point_within_mask(cls, mask_image, center, radius=20):
        x_center, y_center = center
        mask = np.array(mask_image) > 0  # Convert to boolean array
        height, width = mask.shape

        for _ in range(100):  # Try 100 times to find a valid point
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, radius)
            x = x_center + distance * np.cos(angle)
            y = y_center + distance * np.sin(angle)
            x_int, y_int = int(round(x)), int(round(y))
            try:
                if 0 <= x < width and 0 <= y < height and mask[y_int, x_int]:
                    cls.end_point = [x, y]
                    return [x, y]
            except Exception:
                continue
        else:
            raise ValueError("No valid point found within the specified radius.")
