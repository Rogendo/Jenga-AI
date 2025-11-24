# How-to: Configure an Experiment

The `experiment.yaml` file is the single source of truth for any Jenga-AI experiment. It provides a clear and reproducible way to define the model, data, and training parameters. This guide provides a detailed breakdown of all available configuration options.

## Top-Level Structure

Your `experiment.yaml` file is organized into four main sections:

```yaml
project_name: "My_Awesome_Project"

model:
  # ... model configuration ...

tokenizer:
  # ... tokenizer configuration ...

training:
  # ... training configuration ...

tasks:
  # ... a list of task configurations ...
```

---

## `model`

This section defines the architecture of your `MultiTaskModel`.

```yaml
model:
  base_model: "distilbert-base-uncased"
  dropout: 0.1
  fusion:
    type: "attention"
    hidden_size: 768
```

| Key | Type | Description | Required |
| :-- | :--- | :--- | :--- |
| `base_model` | string | The name of a pre-trained model from the [Hugging Face Hub](https://huggingface.co/models). Examples: `bert-base-cased`, `xlm-roberta-base`. | Yes |
| `dropout` | float | The dropout probability for the model's classification heads. | No (defaults to `0.1`) |
| `fusion` | object | Optional configuration for the `AttentionFusion` layer. Use this to create task-specific representations. | No |

### `fusion`

If you include the `fusion` block, you can configure the attention fusion layer.

| Key | Type | Description | Required |
| :-- | :--- | :--- | :--- |
| `type` | string | The type of fusion. Currently, only `"attention"` is supported. | Yes |
| `hidden_size`| int | The hidden size of the fusion layer. **Must match the hidden size of the `base_model`**. | Yes |

---

## `tokenizer`

This section controls the behavior of the Hugging Face `AutoTokenizer`.

```yaml
tokenizer:
  max_length: 256
  padding: "max_length"
  truncation: true
```

| Key | Type | Description | Required |
| :-- | :--- | :--- | :--- |
| `max_length` | int | The maximum sequence length. Texts longer than this will be truncated. | Yes |
| `padding` | string | The padding strategy. `"max_length"` pads all sequences to `max_length`. | No (defaults to `"max_length"`) |
| `truncation`| bool | Whether to truncate sequences. Should generally be `true`. | No (defaults to `true`) |

---

## `training`

This section contains all hyperparameters and settings for the `Trainer`.

```yaml
training:
  output_dir: "./results"
  learning_rate: 2.0e-5
  batch_size: 16
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
    experiment_name: "My_Experiment"
```

| Key | Type | Description | Required |
| :-- | :--- | :--- | :--- |
| `output_dir` | string | Path to the directory where results, logs, and checkpoints will be saved. | Yes |
| `learning_rate` | float | The initial learning rate for the AdamW optimizer. | Yes |
| `batch_size` | int | The number of samples per batch for both training and evaluation. | Yes |
| `num_epochs` | int | The total number of training epochs to perform. | Yes |
| `weight_decay`| float | The weight decay to apply (if not zero). | No (defaults to `0.0`) |
| `warmup_steps`| int | Number of steps for the linear warmup of the learning rate. | No (defaults to `0`) |
| `eval_strategy`| string | When to run evaluation. `"epoch"` or `"steps"`. If `"steps"`, you must also set `eval_steps`. | No (defaults to `"epoch"`) |
| `save_strategy`| string | When to save model checkpoints. `"epoch"` or `"steps"`. If `"steps"`, you must also set `save_steps`. | No (defaults to `"epoch"`) |
| `load_best_model_at_end` | bool | If `true`, the trainer will load the best model (based on `metric_for_best_model`) at the end of training. | No (defaults to `false`) |
| `metric_for_best_model` | string | The metric used to identify the best model. Example: `eval_loss`, `eval_f1`. | Required if `load_best_model_at_end` is `true`. |
| `greater_is_better` | bool | Set to `true` if a higher value of `metric_for_best_model` is better (e.g., F1 score), and `false` if a lower value is better (e.g., loss). | Required if `load_best_model_at_end` is `true`. |
| `early_stopping_patience`| int | Number of evaluations with no improvement after which training will be stopped. | No |
| `logging` | object | Configuration for experiment tracking. | Yes |

### `logging`

| Key | Type | Description | Required |
| :-- | :--- | :--- | :--- |
| `service` | string | The logging service to use. Supported: `"mlflow"`, `"tensorboard"`. | Yes |
| `experiment_name` | string | The name for the experiment run in MLflow or TensorBoard. | Yes |
| `tracking_uri` | string | Optional. The URI for a remote MLflow tracking server. | No |

---

## `tasks`

This is a **list** of one or more task objects. Each object defines a task for the model to learn.

```yaml
tasks:
  - name: "Sentiment"
    type: "single_label_classification"
    data_path: "/path/to/sentiment_data.jsonl"
    heads:
      - name: "sentiment_head"
        num_labels: 2
        weight: 1.0
        label_map:
          "Positive": 0
          "Negative": 1
```

| Key | Type | Description | Required |
| :-- | :--- | :--- | :--- |
| `name` | string | A unique name for the task (e.g., "Sentiment", "ThreatDetection"). | Yes |
| `type` | string | The type of task. Supported types are: `"single_label_classification"`, `"multi_label_classification"`, `"ner"`. | Yes |
| `data_path` | string | The absolute path to the `.jsonl` data file for this task. | Yes |
| `heads` | list | A list of one or more prediction heads for this task. | Yes |

### `heads`

Each task has at least one prediction head.

| Key | Type | Description | Required |
| :-- | :--- | :--- | :--- |
| `name` | string | A unique name for the prediction head. | Yes |
| `num_labels` | int | The number of output labels for this head. For NER, this is the number of entity types. For classification, the number of classes. This value may be automatically updated by the `DataProcessor`. | Yes |
| `weight` | float | The weight of this head's loss in the total loss calculation for the task. Allows you to prioritize certain heads. | No (defaults to `1.0`) |
| `label_map` | dict | Optional. A dictionary mapping string labels from your data file to integer IDs. If not provided, the `DataProcessor` will create this map automatically. | No |
