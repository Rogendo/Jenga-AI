# Logging

The Jenga-NLP framework is integrated with MLflow and TensorBoard for experiment tracking, allowing you to monitor your training process, compare different experiments, and visualize your results.

## How it Works

The `Trainer` class in `multitask_bert/training/trainer.py` is responsible for initializing the logger and logging the metrics during training and evaluation.

At the beginning of the training process, the `Trainer` checks the `logging` section of the `TrainingConfig` to determine which logging service to use. It then initializes the appropriate logger (`SummaryWriter` for TensorBoard or `mlflow` for MLflow).

During the training loop, the `Trainer` logs the training loss at each step. During the evaluation loop, it logs the evaluation metrics for each task and head.

## How to Configure Logging

To configure logging, you need to add a `logging` section to the `training` configuration in your `experiment.yaml` file.

### TensorBoard

To use TensorBoard, set the `service` to `"tensorboard"`:

```yaml
training:
  # ...
  logging:
    service: "tensorboard"
    experiment_name: "JengaAI_MVP"
```

-   `service`: The logging service to use.
-   `experiment_name`: The name of the experiment. The TensorBoard logs will be saved to `output_dir/logs/experiment_name`.

To view the TensorBoard logs, run the following command in your terminal:

```bash
tensorboard --logdir=./unified_results/logs
```

### MLflow

To use MLflow, set the `service` to `"mlflow"`:

```yaml
training:
  # ...
  logging:
    service: "mlflow"
    experiment_name: "JengaAI_MVP"
    tracking_uri: "http://localhost:5000" # Optional
```

-   `service`: The logging service to use.
-   `experiment_name`: The name of the experiment.
-   `tracking_uri`: The tracking URI for the MLflow server (optional). If not provided, MLflow will log to a local `mlruns` directory.

To view the MLflow UI, run the following command in your terminal:

```bash
mlflow ui
```

## `_init_logger` and `_log_metrics`

The `Trainer` class has two key methods for logging:

-   `_init_logger(self)`: This method is called in the constructor of the `Trainer` and is responsible for initializing the logger based on the configuration.

-   `_log_metrics(self, metrics, step, prefix)`: This method is called during the training and evaluation loops to log the metrics. It takes the following arguments:
    -   `metrics`: A dictionary of metrics to log.
    -   `step`: The current step or epoch.
    -   `prefix`: A prefix to add to the metric names (e.g., `"Train"` or `"Eval"`).

By using these methods, the `Trainer` provides a consistent and flexible way to log your experiment results, regardless of which logging service you choose.
