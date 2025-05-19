from pydantic import BaseModel, Field
from typing import List, Optional
from .base import ModificationBase


class RepReMovingModel(ModificationBase):
    name: str = "Object_Removing"
    description: str = "Procedure to remove an object from the image."
    end_point: Optional[List[float]] = Field(
        description="The end position of the center point of the moving object. Be careful not to go beyond the picture boundaries"
    )
