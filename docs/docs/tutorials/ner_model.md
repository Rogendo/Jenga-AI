# Tutorial: Training and Using an NER Model

Named Entity Recognition (NER) is a common NLP task that involves identifying and classifying named entities in text (e.g., persons, organizations, locations).

This tutorial will guide you through training a powerful NER model using Jenga-AI's streamlined workflow, and then using that model for inference.

## 1. The NER Configuration File

We will use the `experiment_ner.yaml` file provided in the `examples/` directory. This file is configured specifically for an NER task.

`examples/experiment_ner.yaml`:
```yaml
project_name: "JengaAI_NER_Experiment"

model:
  base_model: "distilbert-base-uncased"
  dropout: 0.1

tokenizer:
  max_length: 512
  padding: "max_length"
  truncation: true

training:
  output_dir: "./unified_results_ner"
  learning_rate: 2.0e-5
  batch_size: 8
  num_epochs: 5
  weight_decay: 0.01
  warmup_steps: 100
  eval_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "eval_SwahiliNER_loss" 
  greater_is_better: false
  logging:
    service: "mlflow"
    experiment_name: "JengaAI_NER"

tasks:
  - name: "SwahiliNER"
    type: "ner"
    # This path is relative to the project root
    data_path: "examples/ner_synthetic_dataset_v1.jsonl"
    heads:
      - name: "ner_head"
        # The DataProcessor will automatically determine the correct number of labels
        num_labels: 13 
        weight: 1.0
```

### Key Differences for NER:
- **`type: "ner"`**: This tells the framework to use the specialized `NERTask` class, which applies classification at the token level.
- **`data_path`**: The data file (`ner_synthetic_dataset_v1.jsonl`) contains text and a list of `entities` with character offsets, which the `DataProcessor` knows how to handle for NER tasks.

## 2. Train the NER Model

Thanks to the refactored `run_experiment.py` script, training the model is simple and clean. The script uses the `Trainer.from_config` method to handle all the setup boilerplate.

To run the training, execute the following command from the **root of the Jenga-AI project**:

```bash
python examples/run_experiment.py --config examples/experiment_ner.yaml
```

This command will:
1.  Instantiate the `Trainer`, which automatically sets up the tokenizer, data processor, and `MultiTaskModel`.
2.  Process the NER data, aligning entity labels with tokens.
3.  Train the model for 5 epochs, saving the best-performing checkpoint based on evaluation loss.
4.  Save the final model, tokenizer, and configuration to the `./unified_results_ner/best_model/` directory.

## 3. Run Inference with the Trained Model

After training, you can use the `examples/ner_iinference.py` script to load your trained model and make predictions on new text.

This script is a self-contained example that shows all the steps required for inference:
- Loading the saved experiment configuration.
- Re-instantiating the model architecture.
- Loading the saved model weights.
- Tokenizing new text and running it through the model.
- Post-processing the model's output to extract named entities.

`examples/ner_iinference.py`:
```python
import torch
import yaml
import os
# ... other imports

class NERInference:
    def __init__(self, model_dir: str):
        # ... loads config, tokenizer, and model ...
    
    def _load_model(self):
        # ... logic to instantiate model and load weights ...
        
    def predict_entities(self, text: str):
        # ... preprocesses text, runs model, and post-processes logits ...
        
    def _extract_entities(self, ...):
        # ... logic to convert token predictions into entities ...

def main():
    model_dir = "./unified_results_ner"
    ner_inference = NERInference(model_dir)
    
    test_texts = [
        "Hello, I'm Vincent from Dar es Salaam.",
        "Mwangi Kennedy was taken to the Hospital in Nairobi.",
    ]
    
    for text in test_texts:
        print(f"\n📝 Text: \"{text}\"")
        entities = ner_inference.predict_entities(text)
        # ... prints entities ...

if __name__ == "__main__":
    main()
```

### How it Works
The script defines an `NERInference` class that encapsulates all the logic needed to go from a saved model directory to predicted entities. While this approach is more verbose than the planned `InferenceWrapper`, it clearly shows all the steps involved in the process.

To run the inference script, execute the following command from the project root:

```bash
python examples/ner_iinference.py
```

### Expected Output:
```
Loading trained model...
Using device: cpu
Loading config from: ./unified_results_ner/experiment_config.yaml
Loading model from: ./unified_results_ner/best_model
...
📝 Text: "Hello, I'm Vincent from Dar es Salaam."
🔍 Entities found:
   • 'vincent' → PERSON (chars 13-20)
   • 'dar es salaam' → LOCATION (chars 26-39)

📝 Text: "Mwangi Kennedy was taken to the Hospital in Nairobi."
🔍 Entities found:
   • 'mwangi kennedy' → PERSON (chars 0-14)
   • 'hospital' → ORG (chars 35-43)
   • 'nairobi' → LOCATION (chars 47-54)
```

This demonstrates a complete, end-to-end workflow for training and using an NER model within the Jenga-AI framework.