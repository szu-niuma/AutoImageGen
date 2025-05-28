from pydantic import BaseModel, Field


class MethodModels(BaseModel):
    edit_method: str = Field(..., description="The selected editing method")
    edit_target: str = Field(..., description="The target object being edited")
    edit_process: str = Field(..., description="Specific prompt words regarding the edited content")


class ResponseMethod(BaseModel):
    objects: list[MethodModels] = Field(..., description="List of Editing method")

    @classmethod
    def from_list(cls, items: list[dict]) -> "ResponseMethod":
        return cls(objects=[MethodModels(**item) for item in items])
