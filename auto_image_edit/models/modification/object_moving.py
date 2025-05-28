from typing import List
from pydantic import Field
from .base import ModificationBase


class RepMovingModel(ModificationBase):
    name: str = "Object_Moving"
    description: str = "Procedure to move an object to a new location within the image."
    result: str = Field(description="A detailed description of the final appearance of the object after editing")
    end_point: List[float] = Field(
        description="The end position of the center point of the moving object. Be careful not to go beyond the picture boundaries"
    )

    def need_scales(self):
        return ["end_point"]


# from ..editing_method_enum import EditingMethodEnum
# class SaveMoveModel(RepMovingModel):
#     modify_type: ClassVar[EditingMethodEnum] = EditingMethodEnum.OBJECT_MOVING
#     image_path: str = Field(description="Path of the image")
#     mask_path: str = Field(description="Path of the mask")
#     reference_mask_path: Optional[str] = Field(description="Path of the reference image")
