from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
from ..core.config import TaskConfig
import torch.nn as nn # Import torch.nn

@dataclass
class TaskOutput:
    """Base class for task-specific model outputs."""
    loss: float
    logits: Dict[str, Any]

class BaseTask(nn.Module, ABC): # Inherit from nn.Module
    """
    An abstract base class for defining a task (e.g., classification, NER).
    It is initialized from a TaskConfig and defines the task-specific head.
    """
    def __init__(self, config: TaskConfig):
        super().__init__() # Call nn.Module's constructor
        self.config = config # Store the config
        self.name = config.name
        self.type = config.type
        self.heads = nn.ModuleDict() # Initialize heads as a ModuleDict

    @abstractmethod
    def get_forward_output(self, feature: Dict, pooled_output, sequence_output, **kwargs) -> TaskOutput:
        """
        Defines the forward pass for the task-specific head(s).

        Args:
            feature: The input feature for a single example.
            pooled_output: The pooled output from the shared encoder.
            sequence_output: The sequence output from the shared encoder.

        Returns:
            A TaskOutput object containing the loss and logits.
        """
        raise NotImplementedError
