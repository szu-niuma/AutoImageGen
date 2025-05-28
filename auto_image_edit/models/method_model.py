from pydantic import BaseModel, Field


class MethodModels(BaseModel):
    edit_method: str = Field(..., description="The selected editing method")
    edited_object: str = Field(..., description="The specific name of the object to be edited (e.g., airplane, book)")
    edit_prompt: str = Field(
        ...,
        description="A detailed and comprehensive editing prompt that describes the object transformation process with specific implementation details.",
    )


class ResponseMethod(BaseModel):
    editorial_inspiration: list[MethodModels] = Field(..., description="List of editorial inspiration methods")

    @classmethod
    def from_list(cls, items: list[dict]) -> "ResponseMethod":
        return cls(editorial_inspiration=[MethodModels(**item) for item in items])
