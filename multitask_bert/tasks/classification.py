import torch
import torch.nn as nn
from typing import Dict
from .base import BaseTask, TaskOutput
from ..core.config import TaskConfig

class SingleLabelClassificationTask(BaseTask):
    """
    A task for single-label, multi-class classification with one or more heads.
    """
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        # Each head is a separate linear classifier
        for head in config.heads:
            self.heads[head.name] = nn.Linear(768, head.num_labels)

    def get_forward_output(self, feature: Dict, pooled_output: torch.Tensor, **kwargs) -> TaskOutput:
        total_loss = 0
        all_logits = {}
        loss_fct = nn.CrossEntropyLoss()

        # For single-label classification, there's typically only one head
        # and the labels are a single tensor, not a dictionary.
        # We iterate through heads for consistency, but expect only one.
        for head_config in self.config.heads:
            head_name = head_config.name
            head_layer = self.heads[head_name]
            
            logits = head_layer(pooled_output)
            all_logits[head_name] = logits

            # Only calculate loss if labels are provided
            if 'labels' in feature and feature['labels'] is not None:
                labels = feature['labels']
                loss = loss_fct(logits.view(-1, head_config.num_labels), labels.view(-1))
                total_loss += loss * head_config.weight
        
        return TaskOutput(loss=total_loss, logits=all_logits)

class MultiLabelClassificationTask(BaseTask):
    """
    A task for multi-label classification, where each head can predict multiple binary outcomes.
    This is suitable for the QA scoring task.
    """
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        for head in config.heads:
            self.heads[head.name] = nn.Linear(768, head.num_labels)

    def get_forward_output(self, feature: Dict, pooled_output: torch.Tensor, **kwargs) -> TaskOutput:
        total_loss = 0
        all_logits = {}
        # Use BCEWithLogitsLoss for multi-label binary classification
        loss_fct = nn.BCEWithLogitsLoss()

        for head_config in self.config.heads:
            head_name = head_config.name
            head_layer = self.heads[head_name]
            
            logits = head_layer(pooled_output)
            all_logits[head_name] = logits

            # Only calculate loss if labels are provided and the specific head's labels exist
            if 'labels' in feature and feature['labels'] is not None and head_name in feature['labels'] and feature['labels'][head_name] is not None:
                labels = feature['labels'][head_name]
                loss = loss_fct(logits, labels.float())
                total_loss += loss * head_config.weight
        
        return TaskOutput(loss=total_loss, logits=all_logits)