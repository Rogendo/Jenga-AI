# API Reference: Tasks

A "Task" in Jenga-AI represents a specific NLP problem you want the model to solve. Each task is defined as a Python class that inherits from `BaseTask` and encapsulates the logic for its specific prediction heads and loss calculation.

## `BaseTask`

**Class Path:** `multitask_bert.tasks.base.BaseTask`

The `BaseTask` is an abstract class that provides the fundamental structure for all task implementations. It is a `torch.nn.Module` that defines the interface the `MultiTaskModel` uses to interact with a task.

### Key Components

- **`__init__(self, config, hidden_size)`**
    - **Purpose:** The constructor for the task. Your implementation should call `super().__init__()` and then define the task-specific prediction heads (e.g., `nn.Linear` layers) and store them in the `self.heads` `ModuleDict`.
    - **Arguments:**
        - `config (TaskConfig)`: The configuration object for this specific task.
        - `hidden_size (int)`: The hidden size of the encoder's output, which is needed to correctly dimension the prediction heads.

- **`get_forward_output(...)`**
    - **Purpose:** This abstract method must be implemented by every subclass. It defines the task's forward pass, which takes the output from the shared encoder and produces the final logits and loss.
    - **Arguments:**
        - `encoder_outputs`: The output object from the Hugging Face encoder, which contains the `last_hidden_state` and `pooler_output`.
        - `labels`: The labels for the current batch.
    - **Returns:** A dictionary containing:
        - `loss (torch.Tensor)`: The calculated loss for the batch.
        - `logits (Dict[str, torch.Tensor])`: A dictionary mapping each head name to its output logits.

## Supported Task Implementations

Jenga-AI comes with several pre-built task implementations.

### `MultiHeadSingleLabelClassificationTask`

- **Class Path:** `multitask_bert.tasks.classification.MultiHeadSingleLabelClassificationTask`
- **Config `type` string:** `"classification"`
- **Description:** A versatile task for single-label classification. It can have multiple prediction heads, where each head predicts a single class from its own set of labels. For example, one head could predict sentiment (Positive/Negative) while another predicts emotion (Happy/Sad/Angry).
- **Heads:** Uses the `pooled_output` (from the `[CLS]` token) and passes it through one or more `nn.Linear` layers.
- **Loss Function:** `nn.CrossEntropyLoss`.

### `MultiLabelClassificationTask`

- **Class Path:** `multitask_bert.tasks.classification.MultiLabelClassificationTask`
- **Config `type` string:** `"multi_label_classification"`
- **Description:** Used for multi-label problems where each input can be assigned zero or more labels from a set. For example, tagging an article with multiple topics.
- **Heads:** Uses the `pooled_output` and passes it through one or more `nn.Linear` layers.
- **Loss Function:** `nn.BCEWithLogitsLoss`.

### `NERTask`

- **Class Path:** `multitask_bert.tasks.ner.NERTask`
- **Config `type` string:** `"ner"`
- **Description:** Used for Named Entity Recognition. This is a token-level classification task where the goal is to assign a label to each token in the input sequence (e.g., `B-PERSON`, `I-LOCATION`, `O`).
- **Heads:** Uses the `last_hidden_state` (the representation for every token) and passes it through a `nn.Linear` layer to get a logit for each token.
- **Loss Function:** `nn.CrossEntropyLoss`.

See the **[How-to: Add a New Task](guides/adding_a_new_task.md)** guide for a step-by-step walkthrough of how to create your own custom task.
