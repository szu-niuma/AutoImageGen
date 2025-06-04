from enum import Enum

from pydantic import BaseModel, Field


# 定义一个枚举类
class DifficultyLevel(Enum):
    LOW = "Easy"
    MEDIUM = "Moderate"
    HIGH = "Difficult"
    UNDETECTABLE = "Undetectable"


class RepForensicsAccessModel(BaseModel):
    image_desc: str = Field(
        description="Concise summary of the image’s overall structure, key content, and visually salient elements"
    )
    difficulty_level: DifficultyLevel = Field(
        description=(
            "Estimated difficulty level (e.g., Easy, Medium, Hard) for conducting digital forensics on the edited image, "
            "considering factors such as forgery sophistication, visual integration, and detectability"
        )
    )
    analysis_conclusion: str = Field(
        description=(
            "In response to the above comparison results, a conclusive summary is given, emphasizing the core differences found and their significance in image forensics."
        )
    )
