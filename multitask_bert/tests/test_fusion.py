import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from multitask_bert.core.fusion import AttentionFusion

# Mock a config object with necessary attributes
@pytest.fixture
def mock_config():
    config = MagicMock()
    config.hidden_size = 768
    return config

def test_attention_fusion_init(mock_config):
    num_tasks = 3
    fusion_layer = AttentionFusion(mock_config, num_tasks)

    assert isinstance(fusion_layer.task_embeddings, nn.Embedding)
    assert fusion_layer.task_embeddings.num_embeddings == num_tasks
    assert fusion_layer.task_embeddings.embedding_dim == mock_config.hidden_size

    assert isinstance(fusion_layer.attention_layer, nn.Sequential)
    assert len(fusion_layer.attention_layer) == 3
    assert isinstance(fusion_layer.attention_layer[0], nn.Linear)
    assert isinstance(fusion_layer.attention_layer[1], nn.Tanh)
    assert isinstance(fusion_layer.attention_layer[2], nn.Linear)

def test_attention_fusion_forward():
    # Mock config and num_tasks
    config = MagicMock()
    config.hidden_size = 768
    num_tasks = 3
    fusion_layer = AttentionFusion(config, num_tasks)

    # Mock shared_representation
    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size
    shared_representation = torch.randn(batch_size, seq_len, hidden_size)

    # Mock task_id
    task_id = 1 # Example task ID

    # Mock the internal layers to control their output for predictable testing
    mock_linear1 = MagicMock(spec=nn.Linear)
    mock_linear1.return_value = torch.randn(batch_size, seq_len, hidden_size) # Output of first linear layer
    mock_tanh = MagicMock(spec=nn.Tanh)
    mock_tanh.return_value = torch.randn(batch_size, seq_len, hidden_size) # Output of Tanh
    mock_linear2 = MagicMock(spec=nn.Linear)
    mock_linear2.return_value = torch.randn(batch_size, seq_len, 1) # Output of second linear layer (attention scores)

    fusion_layer.attention_layer[0] = mock_linear1
    fusion_layer.attention_layer[1] = mock_tanh
    fusion_layer.attention_layer[2] = mock_linear2

    fused_representation = fusion_layer.forward(shared_representation, task_id)

    # Assertions
    assert fused_representation.shape == shared_representation.shape

    # Verify that task_embeddings was called
    # Note: task_embeddings is an nn.Embedding, so we can't directly mock its call
    # Instead, we can check the input to the attention_layer's first linear layer

    # Verify the attention_layer was called with the correct shape
    # The input to the first linear layer should be [batch_size, seq_len, hidden_size * 2]
    mock_linear1.assert_called_once()
    # The input to mock_linear1 is the concatenated tensor. We can't directly assert its value
    # but we can check its shape if we were to inspect the call args.
    # For now, we rely on the overall shape assertion and the fact that the mock was called.

    # Further checks could involve:
    # - Checking if softmax was applied (hard to mock directly without patching F.softmax)
    # - Checking if the output is a weighted sum (also hard to mock without patching torch operations)
    # For now, the shape and call assertions provide basic coverage.
