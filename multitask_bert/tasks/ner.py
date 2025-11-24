import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base import BaseTask
from ..core.config import TaskConfig
from ..core.registry import register_task

@register_task("ner")
class NERTask(BaseTask):
    """
    A task for Named Entity Recognition (token classification).
    Adapted from the working independent NER script.
    """
    def __init__(self, config: TaskConfig, hidden_size: int): # Added hidden_size
        super().__init__(config, hidden_size) # Pass hidden_size to super
        # Use the same architecture as the working script
        for head in config.heads:
            self.heads[head.name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, head.num_labels) # Use self.hidden_size
            )
            
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights like the original BERT model"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def get_forward_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        labels: Optional[Any],
        encoder_outputs: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        For NER, we use the sequence_output from the encoder.
        """
        total_loss = 0
        all_logits = {}
        
        # Use the same loss function as working script
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # CRITICAL: Add ignore_index

        sequence_output = encoder_outputs.last_hidden_state # Get sequence_output from encoder_outputs

        for head_config in self.config.heads:
            head_name = head_config.name
            head_layer = self.heads[head_name]
            
            logits = head_layer(sequence_output)
            all_logits[head_name] = logits

            # Only calculate loss if labels is not None
            if labels is not None:
                # Reshape for loss calculation - CRITICAL FIX
                active_loss = labels.view(-1) != -100
                active_logits = logits.view(-1, head_config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]

                if active_labels.numel() > 0:
                    loss = loss_fct(active_logits, active_labels)
                    total_loss += loss * head_config.weight
                else:
                    # If no valid labels, use zero loss
                    total_loss += torch.tensor(0.0, device=logits.device, requires_grad=True)

        return {"loss": total_loss, "logits": all_logits}