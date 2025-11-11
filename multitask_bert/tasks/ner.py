import torch
import torch.nn as nn
from typing import Dict
from .base import BaseTask, TaskOutput
from ..core.config import TaskConfig

class NERTask(BaseTask):
    """
    A task for Named Entity Recognition (token classification).
    """
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        # NER usually has a single head, but we build it to be consistent
        for head in config.heads:
            self.heads[head.name] = nn.Linear(768, head.num_labels)

    def get_forward_output(self, feature: Dict, sequence_output: torch.Tensor, **kwargs) -> TaskOutput:
        """
        For NER, we use the sequence_output from the encoder.
        """
        total_loss = 0
        all_logits = {}
        loss_fct = nn.CrossEntropyLoss()

        # Assuming one head for NER, but iterating for consistency
        for head_config in self.config.heads:
            head_name = head_config.name
            head_layer = self.heads[head_name]
            
            logits = head_layer(sequence_output)
            all_logits[head_name] = logits

            # Only calculate loss if labels are provided
            if 'labels' in feature and feature['labels'] is not None:
                labels = feature['labels'] # For NER, labels are a single tensor
                
                # Only keep active parts of the loss
                if 'attention_mask' in feature:
                    active_loss = feature['attention_mask'].view(-1) == 1
                    active_logits = logits.view(-1, head_config.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, head_config.num_labels), labels.view(-1))
                
                total_loss += loss * head_config.weight

        return TaskOutput(loss=total_loss, logits=all_logits)