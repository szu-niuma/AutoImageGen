# parsers/models/editing_method_enum.py

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Type

from pydantic import BaseModel

from .modification import (
    RepContentDragModel,
    RepGenerationModel,
    RepMovingModel,
    RepRecoloringModel,
    RepReMovingModel,
    RepResizingModel,
)


@dataclass(frozen=True)
class EditingMethod:
    """
    Dataclass to hold information about each editing method.
    """

    name: str
    description: str
    format: str
    model: Type[BaseModel]

    def get_editing_method_info(self) -> str:
        return f"Method Name: {self.name} \n Method Description: {self.description}"


def generate_format(model: Type[BaseModel]) -> str:
    """
    Generate the JSON format string based on the Pydantic model's fields.

    Args:
        model (Type[BaseModel]): The Pydantic model class.

    Returns:
        str: A JSON-formatted string representing the editing method's parameters.
    """
    format_dict = {}
    for field_name, field in model.model_fields.items():
        if field_name in {"name", "description"}:
            continue
        format_dict[field_name] = f"<{field}>"
    return json.dumps(format_dict, ensure_ascii=False, indent=4)


def create_editing_method_enum(models: List[Type[BaseModel]]) -> Dict[str, EditingMethod]:
    """
    Dynamically create the EditingMethodEnum based on defined Pydantic models.

    Args:
        models (List[Type[BaseModel]]): A list of Pydantic model classes.

    Returns:
        Dict[str, EditingMethod]: A dictionary mapping method names to EditingMethod instances.
    """
    enum_members: Dict[str, EditingMethod] = {}
    for model in models:
        name = model.model_fields["name"].default
        description = model.model_fields["description"].default
        enum_members[name] = EditingMethod(
            name=name,
            description=description,
            format=generate_format(model),
            model=model,
        )

    return enum_members


# List of all Pydantic models representing different editing methods
models_list = [
    RepMovingModel,
    RepRecoloringModel,
    RepResizingModel,
    RepReMovingModel,
    RepGenerationModel,
    RepContentDragModel,
]

# Create the enum
EditingMethodEnum = create_editing_method_enum(models_list)
