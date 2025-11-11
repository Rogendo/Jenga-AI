import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, config, num_tasks):
        super(AttentionFusion, self).__init__()
        self.config = config
        self.num_tasks = num_tasks
        self.task_embeddings = nn.Embedding(num_tasks, config.hidden_size)
        # The attention mechanism will learn to weigh the shared representation based on the task
        self.attention_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, shared_representation, task_id):
        """
        Fuses the shared representation with a task-specific embedding.

        Args:
            shared_representation (torch.Tensor): The output from the shared encoder.
                                                  Shape: [batch_size, seq_len, hidden_size]
            task_id (int): The ID of the current task.

        Returns:
            torch.Tensor: The task-fused representation.
                          Shape: [batch_size, seq_len, hidden_size]
        """
        # Get the task embedding
        task_embedding = self.task_embeddings(torch.tensor([task_id], device=shared_representation.device))
        
        # Expand task embedding to match the sequence length of the shared representation
        # Shape: [batch_size, seq_len, hidden_size]
        task_embedding_expanded = task_embedding.unsqueeze(0).expand(shared_representation.size(0), shared_representation.size(1), -1)

        # Concatenate the shared representation with the expanded task embedding
        # Shape: [batch_size, seq_len, hidden_size * 2]
        combined_representation = torch.cat([shared_representation, task_embedding_expanded], dim=2)

        # Compute attention scores
        # Shape: [batch_size, seq_len, 1]
        attention_scores = self.attention_layer(combined_representation)
        
        # Apply softmax to get attention weights
        # Shape: [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)

        # Multiply the shared representation by the attention weights to get the fused representation
        # Shape: [batch_size, seq_len, hidden_size]
        fused_representation = shared_representation * attention_weights

        return fused_representation