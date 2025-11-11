# Configuration

The Jenga-NLP framework is designed to be highly configurable, allowing you to define and customize your experiments using a single YAML file. This document provides a detailed explanation of the configuration options available.

## Top-Level Configuration (`ExperimentConfig`)

The main configuration is defined by the `ExperimentConfig` dataclass in `multitask_bert/core/config.py`. It has the following top-level fields:

-   `project_name`: A string that identifies your project.
-   `model`: A `ModelConfig` object that defines the model architecture.
-   `tokenizer`: A `TokenizerConfig` object that defines the tokenizer settings.
-   `training`: A `TrainingConfig` object that defines the training process.
-   `tasks`: A list of `TaskConfig` objects, each defining a specific task.

### `ModelConfig`

The `ModelConfig` dataclass defines the model architecture.

-   `base_model`: The name of the pre-trained transformer model to use from the Hugging Face Hub (e.g., `"distilbert-base-uncased"`).
-   `dropout`: The dropout probability for the model.
-   `fusion`: An optional `FusionConfig` object to enable and configure the attention fusion layer.

#### `FusionConfig`

The `FusionConfig` dataclass enables and configures the attention fusion layer.

-   `type`: The type of fusion to use. Currently, only `"attention"` is supported.
-   `hidden_size`: The hidden size of the fusion layer. This should match the hidden size of the base model.

### `TokenizerConfig`

The `TokenizerConfig` dataclass defines the tokenizer settings.

-   `max_length`: The maximum sequence length for the tokenizer.
-   `padding`: The padding strategy to use (e.g., `"max_length"`).
-   `truncation`: Whether to truncate sequences that are longer than `max_length`.

### `TrainingConfig`

The `TrainingConfig` dataclass defines the training process.

-   `output_dir`: The directory where the training results will be saved.
-   `learning_rate`: The learning rate for the optimizer.
-   `batch_size`: The batch size for training and evaluation.
-   `num_epochs`: The number of training epochs.
-   `weight_decay`: The weight decay for the optimizer.
-   `warmup_steps`: The number of warmup steps for the learning rate scheduler.
-   `eval_strategy`: The evaluation strategy to use (e.g., `"epoch"`).
-   `save_strategy`: The save strategy to use (e.g., `"epoch"`).
-   `load_best_model_at_end`: Whether to load the best model at the end of training.
-   `metric_for_best_model`: The metric to use for determining the best model.
-   `greater_is_better`: Whether a higher value of `metric_for_best_model` is better.
-   `early_stopping_patience`: The number of epochs to wait for improvement before stopping training.
-   `logging`: A `LoggingConfig` object to configure experiment tracking.

#### `LoggingConfig`

The `LoggingConfig` dataclass configures experiment tracking.

-   `service`: The logging service to use. Can be `"tensorboard"` or `"mlflow"`.
-   `experiment_name`: The name of the experiment.
-   `tracking_uri`: The tracking URI for MLflow (optional).

### `TaskConfig`

The `TaskConfig` dataclass defines a specific task.

-   `name`: The name of the task.
-   `type`: The type of the task. Supported types are:
    -   `"single_label_classification"`
    -   `"multi_label_classification"`
    -   `"ner"`
-   `data_path`: The path to the data file for the task.
-   `heads`: A list of `HeadConfig` objects, each defining a prediction head for the task.

#### `HeadConfig`

The `HeadConfig` dataclass defines a prediction head.

-   `name`: The name of the head.
-   `num_labels`: The number of labels for the head.
-   `weight`: The weight of the head's loss in the total loss.

## Example Configuration

Here is an example of a complete `experiment.yaml` file:

```yaml
project_name: "JengaAI_Unified_Framework"

model:
  base_model: "distilbert-base-uncased"
  dropout: 0.1

tokenizer:
  max_length: 256
  padding: "max_length"
  truncation: true

training:
  output_dir: "./unified_results"
  learning_rate: 2.0e-5
  batch_size: 8
  num_epochs: 5
  weight_decay: 0.01
  warmup_steps: 100
  eval_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  early_stopping_patience: 3
  logging:
    service: "mlflow"
    experiment_name: "JengaAI_MVP"

tasks:
  - name: "SentimentClassifier"
    type: "single_label_classification"
    data_path: "examples/sentiment_data.jsonl"
    heads:
      - name: "sentiment_head"
        num_labels: 2
        weight: 1.0

  - name: "SwahiliNER"
    type: "ner"
    data_path: "examples/ner_data.jsonl"
    heads:
      - name: "ner_head"
        num_labels: 13
        weight: 1.0
```
