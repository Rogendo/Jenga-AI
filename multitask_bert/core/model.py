import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig
from typing import List, Dict, Any
from ..tasks.base import BaseTask
from .fusion import AttentionFusion

class MultiTaskModel(PreTrainedModel):
    """
    A generic multi-task model that uses a shared encoder and task-specific heads.
    This version's forward pass is designed to handle a batch from a single task at a time.
    """
    def __init__(self, config, model_config, tasks: List[BaseTask]):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_config.base_model, config=config)
        self.tasks = nn.ModuleList(tasks)
        self.fusion = None
        if model_config.fusion:
            self.fusion = AttentionFusion(config, len(tasks))

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the input embeddings layer of the model.
        """
        return self.encoder.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module):
        """
        Sets the input embeddings layer of the model.
        """
        self.encoder.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_id: int,
        labels: Any = None,
        token_type_ids: torch.Tensor = None, # Keep this for other models that might use it
        **kwargs
    ):
        """
        The forward pass for a single task's batch.

        Args:
            input_ids: Input token ids (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            task_id: The index of the task to run.
            labels: The labels for this task's batch.
            token_type_ids: Token type ids (segment ids).

        Returns:
            A TaskOutput object from the specified task.
        """
        # Pass token_type_ids only if the model supports it
        encoder_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if self.encoder.config.model_type in ["bert", "xlnet", "roberta"]: # Add other models that use token_type_ids
            encoder_inputs['token_type_ids'] = token_type_ids
        
        encoder_outputs = self.encoder(**encoder_inputs, **kwargs)
        
        sequence_output = encoder_outputs.last_hidden_state
        
        if self.fusion:
            sequence_output = self.fusion(sequence_output, task_id)
            
        pooled_output = sequence_output[:, 0]

        task = self.tasks[task_id]
        
        feature = {'labels': labels, 'attention_mask': attention_mask}
        
        task_output = task.get_forward_output(
            feature=feature,
            pooled_output=pooled_output,
            sequence_output=sequence_output
        )
        
        return task_output