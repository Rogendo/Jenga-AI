# Multi-Task Model

The `MultiTaskModel` class in `multitask_bert/core/model.py` is the central component of the Jenga-NLP framework. It is responsible for processing input data, generating representations, and making predictions for multiple tasks simultaneously.

## Architecture

The `MultiTaskModel` is built on top of the Hugging Face `PreTrainedModel` class, which allows it to easily load pre-trained transformer models from the Hugging Face Hub.

The model has the following key components:

1.  **Shared Encoder:** A pre-trained transformer model (e.g., `distilbert-base-uncased`) that acts as a shared encoder for all tasks. This encoder is responsible for generating contextualized representations of the input text.

2.  **Task-Specific Heads:** Each task has one or more prediction heads that are attached to the shared encoder. These heads are responsible for making predictions for their specific task. For example, a classification task might have a simple linear layer as its head, while a named entity recognition (NER) task might have a token-level classification head.

3.  **Attention Fusion Layer (Optional):** The model can optionally include an `AttentionFusion` layer. This layer sits between the shared encoder and the task-specific heads and learns to create task-specific representations from the shared representation.

## `MultiTaskModel` Class

The `MultiTaskModel` class has the following key methods:

-   `__init__(self, config, model_config, tasks)`: The constructor for the model. It takes the following arguments:
    -   `config`: A Hugging Face `PretrainedConfig` object.
    -   `model_config`: A `ModelConfig` object from the experiment configuration.
    -   `tasks`: A list of `BaseTask` objects.

-   `forward(self, input_ids, attention_mask, task_id, labels=None, token_type_ids=None, **kwargs)`: The forward pass for the model. It takes a batch of data for a single task and returns the output of that task.

    -   `input_ids`: The input token IDs.
    -   `attention_mask`: The attention mask.
    -   `task_id`: The ID of the task to run.
    -   `labels`: The labels for the task.
    -   `token_type_ids`: The token type IDs (for models that use them).

## How it Works

1.  **Input Processing:** The `forward` method receives a batch of data for a specific task. The `input_ids` and `attention_mask` are passed to the shared encoder.

2.  **Shared Representation:** The shared encoder processes the input and generates a `last_hidden_state`, which is a sequence of hidden states for each token in the input.

3.  **Attention Fusion (Optional):** If the `AttentionFusion` layer is enabled, it takes the `last_hidden_state` and the `task_id` as input and produces a task-specific representation. This is done by learning an attention mechanism that weighs the shared representation differently for each task.

4.  **Task-Specific Head:** The (potentially fused) representation is then passed to the appropriate task-specific head, which is determined by the `task_id`.

5.  **Output:** The task head produces the final output for the task, which includes the loss and the logits.

## Why this Architecture?

This architecture was chosen for its balance of performance and efficiency.

-   **Parameter Efficiency:** By sharing the encoder across all tasks, the model can learn more general-purpose representations of the language, which can lead to better performance on all tasks. It also significantly reduces the number of parameters that need to be trained, making the model more efficient to train and deploy.

-   **Flexibility:** The use of task-specific heads allows the framework to handle a wide range of NLP tasks, each with its own specific output format.

-   **Extensibility:** The modular design makes it easy to add new tasks or models to the framework without having to modify the core architecture.
