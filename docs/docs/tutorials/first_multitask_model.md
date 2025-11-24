# Tutorial: Training Your First Multi-Task Model

In this tutorial, we will take a deep dive into the Jenga-AI framework by training a single model on two distinct NLP tasks:
1.  **Sentiment Analysis:** A single-label classification task.
2.  **Named Entity Recognition (NER):** A token-level classification task.

This guide will walk you through preparing the data, writing a comprehensive configuration file, and running the training script.

## 1. Project Setup

First, create a new project directory with the following structure. We will use the dummy data files that come with the Jenga-AI repository.

```
jenga-tutorial/
├── config.yaml
└── train.py
```

For this tutorial, we will reference the dummy data files located in the `examples/` directory of the Jenga-AI repository:
- `examples/sentiment_data.jsonl`
- `examples/ner_data.jsonl`

Make sure you know the full path to these files on your system.

## 2. Understanding the Data

Let's look at the format of our two data files.

`sentiment_data.jsonl`: Each line is a JSON object with a `text` and a `label`.
```json
{"text": "I love this framework, it's so easy to use!", "label": "Positive"}
{"text": "This is the worst product I have ever used.", "label": "Negative"}
```

`ner_data.jsonl`: Each line contains `text` and a list of `entities`, where each entity has a `label` and its `start` and `end` character indices.
```json
{"text": "John Doe visited Nairobi last week.", "entities": [{"label": "PERSON", "start": 0, "end": 8}, {"label": "LOCATION", "start": 17, "end": 24}]}
```

## 3. The Experiment Configuration File

The `config.yaml` file is the heart of your experiment. It defines every aspect of the training process. Let's create a detailed configuration.

Create the `config.yaml` file:
```yaml
# config.yaml

project_name: "JengaAI_Tutorial"

# ---------------------------------
# MODEL CONFIGURATION
# Defines the core model architecture.
# ---------------------------------
model:
  base_model: "distilbert-base-uncased"
  dropout: 0.1
  # We can optionally add an attention fusion layer here
  # fusion:
  #   type: "attention"
  #   hidden_size: 768

# ---------------------------------
# TOKENIZER CONFIGURATION
# Settings for the Hugging Face tokenizer.
# ---------------------------------
tokenizer:
  max_length: 256
  padding: "max_length"
  truncation: true

# ---------------------------------
# TRAINING CONFIGURATION
# All hyperparameters and settings for the training loop.
# ---------------------------------
training:
  output_dir: "./tutorial_results"
  learning_rate: 2.0e-5
  batch_size: 8
  num_epochs: 3
  weight_decay: 0.01
  warmup_steps: 100
  eval_strategy: "epoch"      # Evaluate at the end of each epoch
  save_strategy: "epoch"      # Save a checkpoint at the end of each epoch
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss" # Use evaluation loss to find the best model
  greater_is_better: false    # Lower loss is better
  early_stopping_patience: 2  # Stop if eval_loss doesn't improve for 2 epochs
  logging:
    service: "mlflow"         # Use MLflow for experiment tracking
    experiment_name: "JengaAI_Tutorial_MVP"
    # tracking_uri: "http://localhost:5000" # Optional: for a remote MLflow server

# ---------------------------------
# TASK DEFINITIONS
# A list of all tasks the model should learn.
# ---------------------------------
tasks:
  # --- Task 1: Sentiment Analysis ---
  - name: "Sentiment"
    type: "single_label_classification"
    # IMPORTANT: Use the full path to your data file
    data_path: "/path/to/Jenga-AI/examples/sentiment_data.jsonl"
    heads:
      - name: "sentiment_head"
        num_labels: 2 # We will map labels automatically in the DataProcessor
        weight: 1.0   # The weight of this task's loss in the total loss

  # --- Task 2: Named Entity Recognition ---
  - name: "SwahiliNER"
    type: "ner"
    # IMPORTANT: Use the full path to your data file
    data_path: "/path/to/Jenga-AI/examples/ner_data.jsonl"
    heads:
      - name: "ner_head"
        num_labels: 13 # Placeholder, will be updated by the DataProcessor
        weight: 1.5    # Give the NER task slightly more weight
```
**Important:** Remember to replace `/path/to/Jenga-AI/` with the actual absolute path to the project directory on your machine.

## 4. The Training Script

Now, let's create the `train.py` script. This script is a more robust version of the one from the Quickstart guide and is based on the `run_experiment.py` file in the `examples` directory.

Create the `train.py` file:
```python
# train.py
import argparse
import dataclasses
import os
import yaml

from multitask_bert.core.config import load_experiment_config
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.training.trainer import Trainer

def main(config_path: str):
    """
    Main function to run a multi-task experiment from a config file.
    """
    # 1. Load Config from the YAML file
    print("Loading experiment configuration...")
    config = load_experiment_config(config_path)

    # 2. Process Data for all defined tasks
    # The DataProcessor will read the config, find the data files,
    # and prepare them for training and evaluation. It also automatically
    # determines the label maps for you.
    print("Processing data for all tasks...")
    data_processor = DataProcessor(config)
    train_datasets, eval_datasets, updated_config = data_processor.process_data()
    config = updated_config # The config is updated with the correct number of labels

    # 3. Instantiate the Trainer
    # The Trainer class is the main orchestrator. It will automatically
    # build the MultiTaskModel, optimizer, and scheduler based on the config.
    print("Instantiating trainer...")
    trainer = Trainer(
        config=config,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets
    )

    # 4. Start Training
    print("🔥 Starting training...")
    try:
        trainer.train()
        print("🎉 Training complete.")

        # 5. Final Evaluation on the test set
        print("Running final evaluation...")
        final_metrics = trainer.evaluate(test=True) # Use test=True for final eval
        print("Final evaluation metrics:")
        print(final_metrics)

    finally:
        # 6. Close the logger to ensure all data is saved
        trainer.close()
        print("Logger closed.")

    # 7. Save the final, updated config
    output_config_path = os.path.join(config.training.output_dir, "experiment_config.yaml")
    with open(output_config_path, 'w') as f:
        yaml.dump(dataclasses.asdict(config), f, indent=2)
    print(f"✅ Final experiment config saved to: {output_config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Jenga-AI tutorial experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the experiment YAML file."
    )
    args = parser.parse_args()
    main(args.config)
```

## 5. Run the Experiment

You are now ready to train your multi-task model. From your terminal, inside the `jenga-tutorial` directory, run the following command:

```bash
python train.py --config config.yaml
```

The script will:
1.  Load your `config.yaml`.
2.  Process both `sentiment_data.jsonl` and `ner_data.jsonl`.
3.  Build a `distilbert-base-uncased` model with two separate "heads" (one for sentiment, one for NER).
4.  Train the model, alternating between batches of sentiment and NER data.
5.  Evaluate the model's performance on each task at the end of every epoch.
6.  Save the best model, logs, and results into the `tutorial_results` directory.

Congratulations! You have successfully trained a multi-task model using Jenga-AI. You can now explore the output directory and analyze the results in MLflow by running `mlflow ui`.
