import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the root directory to the path
sys.path.insert(0, os.path.abspath('.'))

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

def main():
    print("🧪 Simple Attention Fusion Test")
    print("=" * 50)
    
    # Create a simple config
    class SimpleConfig:
        def __init__(self, hidden_size=768):
            self.hidden_size = hidden_size
    
    config = SimpleConfig(768)
    num_tasks = 2
    
    # Initialize fusion
    fusion = AttentionFusion(config, num_tasks)
    print(f"✅ Fusion initialized for {num_tasks} tasks")
    
    # Create dummy data
    batch_size = 2
    seq_len = 5
    hidden_size = config.hidden_size
    
    shared_repr = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Input shape: {shared_repr.shape}")
    
    # Test different tasks
    print("\n🔍 Testing task-specific fusion:")
    
    for task_id in range(num_tasks):
        fused = fusion(shared_repr, task_id)
        print(f"Task {task_id}: {fused.shape}")
        
        # Show that attention is working
        print(f"  - Input norm: {shared_repr.norm():.4f}")
        print(f"  - Output norm: {fused.norm():.4f}")
        print(f"  - Mean absolute change: {(fused - shared_repr).abs().mean():.4f}")
    
    # Verify different tasks produce different outputs
    fused_0 = fusion(shared_repr, 0)
    fused_1 = fusion(shared_repr, 1)
    
    are_different = not torch.allclose(fused_0, fused_1, atol=1e-6)
    print(f"\n🎯 Different outputs for different tasks: {are_different}")
    
    if are_different:
        print("✅ SUCCESS: Attention fusion is working! Each task gets unique representation.")
    else:
        print("❌ WARNING: Tasks are getting identical representations.")
    
    # Show the architecture
    print(f"\n🏗️  Fusion Architecture:")
    print(f"  - Task embeddings: {fusion.task_embeddings.weight.shape}")
    print(f"  - Attention layer: {fusion.attention_layer}")
    
    print("\n🎉 Test completed successfully!")

if __name__ == "__main__":
    main()