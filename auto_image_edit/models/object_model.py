from pydantic import BaseModel, Field


class ObjectModels(BaseModel):
    label: str = Field(..., description="The label of the object.")
    caption: str = Field(..., description="The caption of the object, using referential expressions.")


class ResponseObject(BaseModel):
    objects: list[ObjectModels] = Field(
        ..., description="List of detected objects with their labels and bounding boxes."
    )

    @classmethod
    def from_list(cls, items: list[dict]) -> "ResponseObject":
        """
        根据一个 dict 列表构造 ResponseObject，
        items 中每个 dict 必须包含 'box_2d' 和 'label' 两个字段。
        """
        return cls(objects=[ObjectModels(**item) for item in items])
