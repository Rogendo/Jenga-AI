import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base import BaseTask
from ..core.config import TaskConfig
from ..core.registry import register_task

@register_task("classification")
class MultiHeadSingleLabelClassificationTask(BaseTask):
    """
    A task for multi-head, single-label, multi-class classification.
    Each head predicts a single label from its own set of classes.
    """
    def __init__(self, config: TaskConfig, hidden_size: int): # Added hidden_size
        super().__init__(config, hidden_size) # Pass hidden_size to super
        # Each head is a separate linear classifier
        for head in config.heads:
            self.heads[head.name] = nn.Linear(self.hidden_size, head.num_labels) # Use self.hidden_size

    def get_forward_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        labels: Optional[Any],
        encoder_outputs: Any,
        **kwargs
    ) -> Dict[str, Any]:
        total_loss = None # Initialize as None
        all_logits = {}
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Use -100 to ignore padding in labels

        pooled_output = encoder_outputs.pooler_output if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None else encoder_outputs.last_hidden_state[:, 0]

        for head_config in self.config.heads:
            head_name = head_config.name
            head_layer = self.heads[head_name]
            
            logits = head_layer(pooled_output)
            all_logits[head_name] = logits

            # Only calculate loss if labels are provided and the specific head's labels exist
            if labels is not None and head_name in labels and labels[head_name] is not None:
                head_labels = labels[head_name]
                # Ensure labels are flattened if they come as (batch_size, 1)
                loss = loss_fct(logits.view(-1, head_config.num_labels), head_labels.view(-1))
                if total_loss is None:
                    total_loss = loss * head_config.weight
                else:
                    total_loss += loss * head_config.weight
        
        # If no loss was calculated (e.g., labels were None), return a zero tensor for loss
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=pooled_output.device, requires_grad=True) # Ensure it's a tensor

        return {"loss": total_loss, "logits": all_logits}

@register_task("multi_label_classification")
class MultiLabelClassificationTask(BaseTask):
    """
    A task for multi-label classification, where each head can predict multiple binary outcomes.
    This is suitable for the QA scoring task.
    """
    def __init__(self, config: TaskConfig, hidden_size: int): # Added hidden_size
        super().__init__(config, hidden_size) # Pass hidden_size to super
        for head in config.heads:
            self.heads[head.name] = nn.Linear(self.hidden_size, head.num_labels) # Use self.hidden_size

    def get_forward_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        labels: Optional[Any],
        encoder_outputs: Any,
        **kwargs
    ) -> Dict[str, Any]:
        total_loss = None # Initialize as None
        all_logits = {}
        # Use BCEWithLogitsLoss for multi-label binary classification
        loss_fct = nn.BCEWithLogitsLoss()

        pooled_output = encoder_outputs.pooler_output if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None else encoder_outputs.last_hidden_state[:, 0]

        for head_config in self.config.heads:
            head_name = head_config.name
            head_layer = self.heads[head_name]
            
            logits = head_layer(pooled_output)
            all_logits[head_name] = logits

            # Only calculate loss if labels are provided and the specific head's labels exist
            if labels is not None and head_name in labels and labels[head_name] is not None:
                head_labels = labels[head_name]
                loss = loss_fct(logits, head_labels.float())
                if total_loss is None:
                    total_loss = loss * head_config.weight
                else:
                    total_loss += loss * head_config.weight
        
        # If no loss was calculated (e.g., labels were None), return a zero tensor for loss
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=pooled_output.device, requires_grad=True) # Ensure it's a tensor

        return {"loss": total_loss, "logits": all_logits}