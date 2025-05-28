from typing import List
from pydantic import BaseModel, Field


class ModificationBase(BaseModel):
    editing_procedure: str = Field(
        description="A detailed description of the object editing process, e.g Add a red flower with water droplets on the grass."
    )
    edited_results: str = Field(
        description="A comprehensive description of the visual effects and changes in the image after editing. Include details about the object's appearance, lighting, and integration with the environment."
    )
