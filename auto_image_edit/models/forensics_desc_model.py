from pydantic import BaseModel, Field


class ForensicsDescModel(BaseModel):
    description: str = Field(
        ..., description="A detailed description of the forensic analysis performed on the edited image, including any anomalies or findings."
    )
