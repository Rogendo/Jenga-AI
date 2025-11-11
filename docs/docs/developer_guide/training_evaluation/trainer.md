# Trainer

The `Trainer` class in `multitask_bert/training/trainer.py` is responsible for orchestrating the entire training and evaluation process. It brings together the model, data, and configuration to train the multi-task model and evaluate its performance.

## Key Responsibilities

The `Trainer` class has the following key responsibilities:

-   **Dataloader Creation:** It creates task-specific dataloaders for training and evaluation.
-   **Optimizer and Scheduler:** It creates the optimizer (AdamW) and the learning rate scheduler.
-   **Training Loop:** It implements the main training loop, which iterates over the data and updates the model's weights.
-   **Evaluation Loop:** It implements the evaluation loop, which calculates the metrics for each task.
-   **Logging:** It logs the training and evaluation metrics to MLflow or TensorBoard.
-   **Early Stopping:** It implements early stopping to prevent overfitting.

## Training Process

The training process is designed to handle multiple tasks simultaneously using a round-robin approach.

1.  **Dataloader Iterators:** At the beginning of each epoch, the `Trainer` creates iterators for each task's dataloader.

2.  **Round-Robin Sampling:** The `Trainer` then iterates through the tasks in a round-robin fashion, sampling one batch from each task's dataloader at a time.

3.  **Forward Pass:** For each batch, the `Trainer` performs a forward pass through the `MultiTaskModel`, providing the `input_ids`, `attention_mask`, `labels`, and `task_id`.

4.  **Backward Pass:** The loss from the forward pass is then used to perform a backward pass and update the model's weights.

5.  **Logging:** The training loss is logged to MLflow or TensorBoard at each step.

This process continues until all dataloaders are exhausted.

## Evaluation Process

The evaluation process is performed at the end of each epoch (if `eval_strategy` is set to `"epoch"`).

1.  **Evaluation Mode:** The model is first set to evaluation mode (`model.eval()`).

2.  **Iterate Through Tasks:** The `Trainer` then iterates through each task's evaluation dataloader.

3.  **Collect Predictions and Labels:** For each batch, the `Trainer` performs a forward pass and collects the predictions and labels for each head of the task.

4.  **Compute Metrics:** Once all batches have been processed, the `Trainer` computes the metrics for each head using the appropriate metric function (e.g., `compute_classification_metrics`, `compute_ner_metrics`).

5.  **Log Metrics:** The evaluation metrics are then logged to MLflow or TensorBoard.

## `Trainer` Class

The `Trainer` class has the following key methods:

-   `__init__(self, config, model, tokenizer, train_datasets, eval_datasets)`: The constructor for the `Trainer`. It takes the following arguments:
    -   `config`: The `ExperimentConfig` object.
    -   `model`: The `MultiTaskModel` object.
    -   `tokenizer`: The tokenizer object.
    -   `train_datasets`: A dictionary of training datasets, where the keys are task names.
    -   `eval_datasets`: A dictionary of evaluation datasets, where the keys are task names.

-   `train(self)`: This method starts the training process.

-   `evaluate(self)`: This method performs the evaluation process.

-   `close(self)`: This method closes the logger.
