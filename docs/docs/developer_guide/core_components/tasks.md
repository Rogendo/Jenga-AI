# Tasks

In the Jenga-NLP framework, a "task" represents a specific NLP problem that you want to solve, such as text classification, named entity recognition (NER), or question answering. Each task is defined by a `BaseTask` subclass, which encapsulates the task-specific logic, including the prediction heads, loss functions, and forward pass.

## The `BaseTask` Class

The `BaseTask` class in `multitask_bert/tasks/base.py` is an abstract base class that all other task classes must inherit from. It defines the following key methods:

-   `__init__(self, config)`: The constructor for the task. It takes a `TaskConfig` object from the experiment configuration.

-   `get_forward_output(self, feature, pooled_output, sequence_output)`: This method defines the forward pass for the task. It takes the following arguments:
    -   `feature`: A dictionary containing the labels and attention mask for the task.
    -   `pooled_output`: The pooled output from the shared encoder (usually the hidden state of the `[CLS]` token).
    -   `sequence_output`: The sequence output from the shared encoder (the hidden states for all tokens).

    It should return a `TaskOutput` object, which is a dataclass that contains the loss and the logits for the task.

## Supported Task Types

The Jenga-NLP framework currently supports the following task types:

### `SingleLabelClassificationTask`

-   **Class:** `multitask_bert.tasks.classification.SingleLabelClassificationTask`
-   **Type String:** `"single_label_classification"`
-   **Description:** This task is used for single-label text classification problems, where each input is assigned to one of several mutually exclusive classes.
-   **Heads:** It uses a single `nn.Linear` layer as its prediction head.
-   **Loss Function:** `nn.CrossEntropyLoss`

### `MultiLabelClassificationTask`

-   **Class:** `multitask_bert.tasks.classification.MultiLabelClassificationTask`
-   **Type String:** `"multi_label_classification"`
-   **Description:** This task is used for multi-label text classification problems, where each input can be assigned to multiple classes simultaneously.
-   **Heads:** It can have multiple prediction heads, one for each label. Each head is a `nn.Linear` layer.
-   **Loss Function:** `nn.BCEWithLogitsLoss`

### `NERTask`

-   **Class:** `multitask_bert.tasks.ner.NERTask`
-   **Type String:** `"ner"`
-   **Description:** This task is used for named entity recognition, where the goal is to identify and classify named entities in a text.
-   **Heads:** It uses a token-level classification head, which is a `nn.Linear` layer that is applied to each token's hidden state.
-   **Loss Function:** `nn.CrossEntropyLoss`

## How to Add a New Task

To add a new task to the framework, you need to do the following:

1.  **Create a new task class:** Create a new Python file in the `multitask_bert/tasks/` directory and define a new class that inherits from `BaseTask`.

2.  **Implement the `__init__` method:** In the constructor of your new task class, you need to define the task-specific prediction heads.

3.  **Implement the `get_forward_output` method:** This method should define the forward pass for your task. It should take the `pooled_output` and `sequence_output` from the shared encoder and pass them through the prediction heads to generate the logits. It should also calculate the loss and return a `TaskOutput` object.

4.  **Update `get_task_class`:** In `examples/run_experiment.py`, update the `get_task_class` function to map your new task's type string to your new task class.

5.  **Update the configuration:** In your `experiment.yaml` file, add a new task configuration with the `type` set to your new task's type string.
