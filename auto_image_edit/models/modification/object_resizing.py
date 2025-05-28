from pydantic import Field
from .base import ModificationBase


class RepResizingModel(ModificationBase):
    name: str = "Object_Resizing"
    description: str = "Procedure to resize an object within the image."
    resizing_scale: float = Field(description="Resizing Scaling")
