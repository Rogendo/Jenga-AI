# Attention Fusion

The Jenga-NLP framework includes an optional `AttentionFusion` layer that can be used to learn task-specific representations from the shared encoder. This can lead to improved performance in multi-task learning scenarios, as it allows the model to learn to focus on different aspects of the shared representation for each task.

## How it Works

The `AttentionFusion` layer is a simple yet effective mechanism for creating task-specific representations. It works as follows:

1.  **Task Embeddings:** The `AttentionFusion` layer maintains a set of task embeddings, one for each task. These embeddings are learned during training and capture the unique characteristics of each task.

2.  **Concatenation:** For a given task, the corresponding task embedding is concatenated with the shared representation from the encoder at each token position.

3.  **Attention Mechanism:** The concatenated representation is then passed through a small feed-forward neural network (the "attention layer") to compute an attention score for each token. These scores are then passed through a softmax function to produce a set of attention weights.

4.  **Fused Representation:** The shared representation is then multiplied by the attention weights to produce the final task-specific (or "fused") representation.

This process is illustrated in the following diagram:

```
+-----------------------+
| Shared Representation |
| (from encoder)        |
+-----------------------+
           |
           v
+-----------------------+      +----------------+
|      Concatenate      | <--- | Task Embedding |
+-----------------------+      +----------------+
           |
           v
+-----------------------+
|    Attention Layer    |
|   (Linear + Tanh)     |
+-----------------------+
           |
           v
+-----------------------+
|        Softmax        |
+-----------------------+
           |
           v
+-----------------------+
|   Attention Weights   |
+-----------------------+
           |
           v
+-----------------------+
|      Multiplication   |
+-----------------------+
           |
           v
+-----------------------+
| Fused Representation  |
+-----------------------+
```

## `AttentionFusion` Class

The `AttentionFusion` class in `multitask_bert/core/fusion.py` implements this mechanism. It has the following key methods:

-   `__init__(self, config, num_tasks)`: The constructor for the fusion layer. It takes the following arguments:
    -   `config`: A Hugging Face `PretrainedConfig` object.
    -   `num_tasks`: The total number of tasks.

-   `forward(self, shared_representation, task_id)`: The forward pass for the fusion layer. It takes the following arguments:
    -   `shared_representation`: The output from the shared encoder.
    -   `task_id`: The ID of the current task.

    It returns the fused representation.

## How to Use It

To use the `AttentionFusion` layer, you need to enable it in your `experiment.yaml` file by adding a `fusion` section to the `model` configuration:

```yaml
model:
  base_model: "distilbert-base-uncased"
  dropout: 0.1
  fusion:
    type: "attention"
    hidden_size: 768
```

-   `type`: The type of fusion to use. Currently, only `"attention"` is supported.
-   `hidden_size`: The hidden size of the fusion layer. This should match the hidden size of the base model.

When the `fusion` section is present in the configuration, the `MultiTaskModel` will automatically instantiate the `AttentionFusion` layer and use it to create task-specific representations before passing them to the task heads.
