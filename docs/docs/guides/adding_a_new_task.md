# How-to: Add a New Task

Jenga-AI is designed to be extensible. One of the most common ways to extend the framework is by adding a new task type. This guide will walk you through the process of creating a custom task.

As an example, we will create a `RegressionTask`, which predicts a single continuous value (e.g., a score from 1.0 to 5.0).

## 1. Understand the `BaseTask` Class

Every task in Jenga-AI must inherit from the `BaseTask` class, which is found in `multitask_bert/tasks/base.py`. This class provides the fundamental structure that the framework expects.

The two key methods you need to implement are:
- `__init__(self, config, hidden_size)`: The constructor where you define the task-specific prediction heads.
- `get_forward_output(...)`: The method that defines the forward pass for your task, taking the output from the shared encoder and calculating the loss and logits.

## 2. Create the Task File

First, create a new file in the `multitask_bert/tasks/` directory. Let's call it `regression.py`.

```bash
touch multitask_bert/tasks/regression.py
```

## 3. Implement the Custom Task Class

Now, let's implement our `RegressionTask` inside `multitask_bert/tasks/regression.py`.

```python
# multitask_bert/tasks/regression.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .base import BaseTask
from ..core.config import TaskConfig

class RegressionTask(BaseTask):
    """
    A task for predicting a single continuous value for one or more heads.
    """
    def __init__(self, config: TaskConfig, hidden_size: int):
        # 1. Call the parent constructor
        super().__init__(config, hidden_size)

        # 2. Define the prediction heads
        # For regression, each head is a linear layer that outputs a single value.
        for head in config.heads:
            self.heads[head.name] = nn.Linear(self.hidden_size, 1)

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
        Defines the forward pass and loss calculation for the regression task.
        """
        total_loss = None
        all_logits = {}

        # 3. Define the loss function
        # Mean Squared Error is a common choice for regression.
        loss_fct = nn.MSELoss()

        # Use the pooled output (representation of the [CLS] token)
        pooled_output = encoder_outputs.pooler_output if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None else encoder_outputs.last_hidden_state[:, 0]

        for head_config in self.config.heads:
            head_name = head_config.name
            head_layer = self.heads[head_name]

            # The output of the head is our predicted value (logit)
            # We squeeze it to remove the last dimension (e.g., from [batch_size, 1] to [batch_size])
            logits = head_layer(pooled_output).squeeze(-1)
            all_logits[head_name] = logits

            # 4. Calculate the loss
            if labels is not None and head_name in labels and labels[head_name] is not None:
                head_labels = labels[head_name].float() # Ensure labels are floats
                loss = loss_fct(logits, head_labels)

                if total_loss is None:
                    total_loss = loss * head_config.weight
                else:
                    total_loss += loss * head_config.weight

        # If no labels were provided, return a zero tensor for the loss
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=pooled_output.device, requires_grad=True)

        return {"loss": total_loss, "logits": all_logits}
```

## 4. Register the New Task

To make the framework aware of your new task, you need to add it to the `get_task_class` function in `examples/run_experiment.py` (or your main training script).

```python
# In examples/run_experiment.py

# Import your new task at the top of the file
from multitask_bert.tasks.regression import RegressionTask

def get_task_class(task_type: str) -> BaseTask:
    """Maps a task type string to its corresponding class."""
    if task_type == "classification":
        return MultiHeadSingleLabelClassificationTask
    elif task_type == "multi_label_classification":
        return MultiLabelClassificationTask
    elif task_type == "ner":
        return NERTask
    # Add your new task here
    elif task_type == "regression":
        return RegressionTask
    else:
        raise ValueError(f"Unknown task type: {task_type}")

```
> **Note:** In the future, this manual step will be replaced by an automatic task registry (see Issue #3.4 in `PROJECT_ROADMAP.md`).

## 5. Use the New Task in a Configuration

You can now use your new task in an `experiment.yaml` file by setting the `type` to `"regression"`.

```yaml
# In your experiment.yaml

tasks:
  - name: "ReviewScore"
    type: "regression" # Use the new task type
    data_path: "/path/to/your/regression_data.jsonl"
    heads:
      - name: "score_head"
        # num_labels is not strictly needed here as the head outputs 1 value,
        # but it can be useful for consistency.
        num_labels: 1
        weight: 1.0
```

Your data file (`regression_data.jsonl`) should contain a numeric label:
```json
{"text": "This was an amazing experience, 5 stars!", "label": 5.0}
{"text": "It was okay, but not great.", "label": 2.5}
```

That's it! You have successfully created, registered, and configured a new custom task in Jenga-AI.
