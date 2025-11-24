# API Reference: Core Components

This section provides a detailed reference for the core architectural components of the Jenga-AI framework.

## `ExperimentConfig`

The `ExperimentConfig` is a dataclass that holds the entire configuration for a training run. It's loaded from your `experiment.yaml` file and serves as the single source of truth for the `Trainer`.

See the **[How-to: Configure an Experiment](guides/configuration.md)** guide for a full breakdown of the YAML structure.

### Key Dataclasses

- **`ExperimentConfig`**: The top-level container.
    - `project_name: str`
    - `model: ModelConfig`
    - `tokenizer: TokenizerConfig`
    - `training: TrainingConfig`
    - `tasks: List[TaskConfig]`
- **`ModelConfig`**: Defines the model architecture.
    - `base_model: str`: The Hugging Face model to use as the encoder.
    - `dropout: float`: Dropout rate for the prediction heads.
    - `fusion: Optional[FusionConfig]`: Optional configuration for the attention fusion layer.
- **`TaskConfig`**: Defines a single task.
    - `name: str`: A unique name for the task.
    - `type: str`: The task type (e.g., `"single_label_classification"`, `"ner"`).
    - `data_path: str`: Path to the data file.
    - `heads: List[HeadConfig]`: A list of prediction heads for this task.
- **`HeadConfig`**: Defines a single prediction head.
    - `name: str`: A unique name for the head.
    - `num_labels: int`: The number of output labels for this head.
    - `weight: float`: The weight of this head's loss in the total loss calculation.

---

## `MultiTaskModel`

**Class Path:** `multitask_bert.core.model.MultiTaskModel`

The `MultiTaskModel` is the heart of the framework. It is a PyTorch module that wraps a shared transformer encoder (e.g., BERT, RoBERTa) and manages a set of task-specific prediction heads.

### Architecture

1.  **Shared Encoder:** A single pre-trained transformer model processes the input text and generates contextualized embeddings. This encoder is shared across all tasks, allowing the model to learn rich, general-purpose language representations.
2.  **Task-Specific Heads:** Each task defined in your configuration has its own set of prediction heads. These are lightweight neural network layers (usually `nn.Linear`) that take the encoder's output and produce the final logits for that specific task.
3.  **Single-Task Forward Pass:** The model's `forward` method is designed to process a batch of data for **one specific task at a time**. The `Trainer` manages the process of alternating between different tasks during training.

### Key Methods

- **`__init__(self, config, model_config, task_configs)`**
    - **Purpose:** Initializes the model. It loads the pre-trained encoder and dynamically creates the required task heads based on the provided configurations.
    - **Arguments:**
        - `config`: The Hugging Face `AutoConfig` for the base model.
        - `model_config (ModelConfig)`: The model configuration from your `experiment.yaml`.
        - `task_configs (List[TaskConfig])`: A list of task configurations.

- **`forward(self, input_ids, attention_mask, task_id, labels=None, ...)`**
    - **Purpose:** Performs a forward pass for a single batch of data.
    - **Arguments:**
        - `input_ids (torch.Tensor)`: The input token IDs for the batch.
        - `attention_mask (torch.Tensor)`: The attention mask for the batch.
        - `task_id (int)`: The integer index of the task to perform the forward pass for.
        - `labels (Any)`: The corresponding labels for the batch.
    - **Returns:** A dictionary containing the `loss` and `logits` for the specified task.

---

## `AttentionFusion`

**Class Path:** `multitask_bert.core.fusion.AttentionFusion`

The `AttentionFusion` layer is an optional but powerful component that can be enabled to improve multi-task learning. It creates task-specific representations by learning to re-weigh the shared encoder's output for each task.

### How It Works

1.  **Task Embeddings:** The layer maintains a learnable embedding vector for each task.
2.  **Attention Mechanism:** For a given task, its embedding is combined with the shared representation from the encoder. This combination is passed through a small neural network that computes an "attention score" for each token in the sequence.
3.  **Fused Representation:** The attention scores are used to weigh the importance of each token's representation. The result is a new, "fused" representation that is tailored to the specific requirements of the current task before being passed to the prediction head.

### How to Use

To enable the `AttentionFusion` layer, add a `fusion` block to the `model` section of your `experiment.yaml`:

```yaml
model:
  base_model: "distilbert-base-uncased"
  fusion:
    type: "attention"
    hidden_size: 768 # IMPORTANT: Must match the hidden size of your base_model
```
