from pydantic import BaseModel, Field


class DetectionRes(BaseModel):
    result: str = Field(
        default="Answer real or fake. The answer to whether the image has been tampered with.", required=True
    )
    score: float = Field(
        "The scale ranges from 0 to 1, indicating the likelihood that you believe the image is fake. The closer the value is to 0, the less likely it is to be fake, while the closer it is to 1, the more likely it is to be fake."
    )
    reason_for_thinking: str = Field(default="The reason for thinking this way.", required=True)
