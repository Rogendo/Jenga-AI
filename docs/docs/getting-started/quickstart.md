# Quickstart

Let's train your first multi-task model with Jenga-AI in just a few minutes. In this guide, we will train a single model to perform two tasks simultaneously:
1.  **Sentiment Classification:** Classifying text as "Positive" or "Negative".
2.  **Threat Detection:** A simple binary classification to identify if a text contains a potential threat.

## 1. Project Structure

First, create a new project directory and set up the following file structure:

```
jenga-quickstart/
├── data/
│   ├── sentiment_data.jsonl
│   └── threat_data.jsonl
├── experiment.yaml
└── run.py
```

## 2. Create Your Data Files

Create the two data files inside the `data/` directory.

`data/sentiment_data.jsonl`:
```json
{"text": "I love this framework, it's so easy to use!", "label": "Positive"}
{"text": "The documentation is clear and helpful.", "label": "Positive"}
{"text": "I encountered an error and it was frustrating.", "label": "Negative"}
{"text": "This is the worst product I have ever used.", "label": "Negative"}
```

`data/threat_data.jsonl`:
```json
{"text": "Remember to submit the weekly security report.", "label": "non-threat"}
{"text": "The system is under attack, we need to respond now!", "label": "threat"}
{"text": "I will shut down the entire network if my demands are not met.", "label": "threat"}
{"text": "Let's have a team meeting tomorrow at 10 AM.", "label": "non-threat"}
```

## 3. Define Your Experiment

Now, create the `experiment.yaml` file. This file tells Jenga-AI everything it needs to know about our model, tasks, and training process.

`experiment.yaml`:
```yaml
project_name: "JengaAI_Quickstart"

model:
  base_model: "distilbert-base-uncased" # A small, fast model for quick testing
  dropout: 0.1

tokenizer:
  max_length: 128
  padding: "max_length"
  truncation: true

training:
  output_dir: "./quickstart_results"
  learning_rate: 2.0e-5
  batch_size: 2
  num_epochs: 1 # Just one epoch for a quick run
  weight_decay: 0.01
  eval_strategy: "epoch"
  logging:
    service: "tensorboard" # Logs will be saved to output_dir/logs
    experiment_name: "Quickstart_Run"

tasks:
  # Task 1: Sentiment Analysis
  - name: "Sentiment"
    type: "single_label_classification"
    data_path: "data/sentiment_data.jsonl"
    heads:
      - name: "sentiment_head"
        num_labels: 2 # Positive, Negative
        label_map:
          "Positive": 0
          "Negative": 1
        weight: 1.0

  # Task 2: Threat Detection
  - name: "Threat"
    type: "single_label_classification"
    data_path: "data/threat_data.jsonl"
    heads:
      - name: "threat_head"
        num_labels: 2 # threat, non-threat
        label_map:
          "threat": 0
          "non-threat": 1
        weight: 1.0
```

## 4. Create the Training Script

Finally, create the `run.py` script. This is the code that will load your configuration and start the training process.

`run.py`:
```python
from multitask_bert.core.config import ExperimentConfig
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.training.trainer import Trainer

def main():
    print("🚀 Starting Jenga-AI Quickstart...")

    # 1. Load experiment configuration from YAML
    config = ExperimentConfig.from_yaml("experiment.yaml")
    print("✅ Configuration loaded.")

    # 2. Process the data for all tasks
    data_processor = DataProcessor(config)
    train_datasets, eval_datasets = data_processor.process_data()
    print(f"✅ Data processed for tasks: {[task.name for task in config.tasks]}")

    # 3. Initialize the Trainer
    # The trainer will automatically build the model based on the config
    trainer = Trainer(
        config=config,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
    )
    print("✅ Trainer initialized.")

    # 4. Start training!
    print("🔥 Starting training...")
    trainer.train()
    print("🎉 Training complete!")
    print(f"✨ Your model and results are saved in: {config.training.output_dir}")

if __name__ == "__main__":
    main()
```

## 5. Run the Training

Make sure you have installed Jenga-AI and are in your activated virtual environment. Then, run the script from your terminal:

```bash
python run.py
```

That's it! You've just trained your first multi-task model. You can inspect the results and logs in the `quickstart_results` directory.

Now that you've seen the basics, you can explore the **[Tutorials](tutorials/first_multitask_model.md)** for more in-depth examples.
