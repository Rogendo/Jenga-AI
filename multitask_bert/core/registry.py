# multitask_bert/core/registry.py

from typing import Dict, Type
from ..tasks.base import BaseTask

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {}

def register_task(name: str):
    """A decorator to register a new task class."""
    def decorator(cls: Type[BaseTask]) -> Type[BaseTask]:
        if name in TASK_REGISTRY:
            raise ValueError(f"Task '{name}' is already registered.")
        TASK_REGISTRY[name] = cls
        return cls
    return decorator
