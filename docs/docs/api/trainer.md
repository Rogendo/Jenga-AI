# API Reference: Trainer

The `Trainer` class is the main engine that orchestrates the training and evaluation of your `MultiTaskModel`. It brings together your configuration, data, and model to handle the entire experiment lifecycle.

**Class Path:** `multitask_bert.training.trainer.Trainer`

## Role of the Trainer

The `Trainer` is responsible for:
- **Creating DataLoaders:** It builds PyTorch `DataLoader` instances for each task, using task-specific collate functions to handle batching correctly.
- **Setting up Optimization:** It initializes the AdamW optimizer and a linear learning rate scheduler with warmup.
- **Executing the Training Loop:** It runs the main training loop, iterating over epochs and feeding batches of data to the model.
- **Managing Multi-Tasking:** It uses a round-robin scheduling approach to alternate between batches from different tasks, ensuring the model trains on all tasks in each epoch.
- **Running Evaluation:** It performs evaluation at specified intervals, calculates metrics for each task head, and logs the results.
- **Handling Checkpointing & Early Stopping:** It saves the best model based on a chosen metric and can stop training early if performance stagnates.
- **Logging:** It integrates with MLflow and TensorBoard to log metrics, parameters, and artifacts.

## Key Methods

- **`__init__(self, config, model, tokenizer, train_datasets, eval_datasets)`**
    - **Purpose:** Initializes the `Trainer` instance.
    - **Arguments:**
        - `config (ExperimentConfig)`: The complete experiment configuration object.
        - `model (MultiTaskModel)`: The instantiated multi-task model.
        - `tokenizer`: The Hugging Face tokenizer.
        - `train_datasets (Dict[str, Dataset])`: A dictionary mapping task names to their training `Dataset` objects.
        - `eval_datasets (Dict[str, Dataset])`: A dictionary mapping task names to their evaluation `Dataset` objects.
    - **Note:** In typical usage, you will use the `Trainer.from_config()` classmethod, which handles the initialization of the model, tokenizer, and data processor for you.

- **`train(self)`**
    - **Purpose:** Starts the main training loop.
    - **Process:**
        1.  Initializes the optimizer and scheduler.
        2.  Iterates through the number of epochs defined in `training.num_epochs`.
        3.  In each epoch, it creates iterators for each task's `DataLoader`.
        4.  It then cycles through the tasks in a round-robin fashion, fetching one batch from each task, performing a forward and backward pass, and updating the model weights.
        5.  Logs the training loss at each step.
        6.  At the end of an epoch (or at specified step intervals), it calls `evaluate()`.
        7.  Handles model checkpointing and early stopping logic.

- **`evaluate(self, test: bool = False) -> Dict[str, float]`**
    - **Purpose:** Evaluates the model's performance on the evaluation (or test) dataset.
    - **Arguments:**
        - `test (bool)`: If `False` (default), uses the evaluation dataloaders. If `True`, it will use the test dataloaders (assuming they were created by the `DataProcessor`).
    - **Process:**
        1.  Sets the model to evaluation mode (`model.eval()`).
        2.  Iterates through the dataloader for each task.
        3.  Collects all predictions and labels for each task head.
        4.  Computes the relevant metrics (e.g., F1, accuracy, precision) for each head using the utility functions in `multitask_bert.utils.metrics`.
        5.  Aggregates all metrics into a single dictionary.
    - **Returns:** A dictionary containing all calculated metrics, prefixed with `eval_`. For example: `{'eval_Sentiment_sentiment_head_f1': 0.95, 'eval_SwahiliNER_ner_head_f1': 0.89}`.

- **`close(self)`**
    - **Purpose:** Gracefully shuts down the logger (e.g., `mlflow.end_run()` or `SummaryWriter.close()`).
    - **Details:** This method should be called at the end of your script, typically in a `finally` block, to ensure that all logs are saved correctly.
