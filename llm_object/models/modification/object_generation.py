from typing import List
from pydantic import BaseModel, Field
from .base import ModificationBase


class RepGenerationModel(ModificationBase):
    name: str = "Object_Generation"
    description: str = "Procedure to generate a new object within the image."
    end_point: List[float] = Field(
        description="Coordinates representing the final position of the center point of the generated object within the image. Ensure these coordinates are within the image boundaries."
    )
