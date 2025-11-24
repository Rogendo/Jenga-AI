# How-to: Deploy a Model

A trained model is only useful if it can be deployed to make predictions on new data. This guide outlines the vision for Jenga-AI's deployment and inference tools.

!!! warning "Under Development"
    The deployment and inference tools described here are part of our future roadmap and are **not yet implemented**. You can track the progress of this feature in the `PROJECT_ROADMAP.md` file under **Milestone 2**.

## The Vision: From Training to API in Minutes

Our goal is to make the transition from a trained model to a production-ready API as seamless as possible. The planned workflow will look like this:

### 1. Export the Model

After training is complete, you will be able to run an export script that packages your `MultiTaskModel` and all its required components into a single, portable artifact.

This artifact will contain:
- The trained model weights.
- The `ExperimentConfig` file, containing all metadata about the tasks and heads.
- The Hugging Face `Tokenizer` files.

**Planned Usage:**
```bash
python -m multitask_bert.deployment.export \
    --checkpoint_dir ./tutorial_results/checkpoint-123 \
    --output_dir ./exported_models/my_multitask_model
```

### 2. Run Inference with a Unified Wrapper

Once exported, you will be able to load the model artifact using a simple `InferenceWrapper` class. This class will provide a clean interface to make predictions for any of the tasks the model was trained on.

**Planned Usage:**
```python
from multitask_bert.deployment import InferenceWrapper

# Load the exported model artifact
model = InferenceWrapper(model_path="./exported_models/my_multitask_model")

# Predict sentiment for a piece of text
sentiment_result = model.predict(
    text="Jenga-AI is an incredibly useful framework!",
    task_name="Sentiment"
)
# Expected output: {'sentiment_head': {'label': 'Positive', 'score': 0.98}}

# Predict named entities from another piece of text
ner_result = model.predict(
    text="John Doe is traveling to Nairobi.",
    task_name="SwahiliNER"
)
# Expected output: {'ner_head': [{'entity': 'PERSON', 'word': 'John Doe'}, {'entity': 'LOCATION', 'word': 'Nairobi'}]}
```

### 3. Deploy as a REST API

The final step is to serve the model as a REST API. We plan to provide a pre-built FastAPI application that can be launched with a single command.

**Planned Usage:**
```bash
# This command will start a Uvicorn server with the FastAPI app
python -m api.main --model_path ./exported_models/my_multitask_model
```

You would then be able to send `POST` requests to the API to get predictions in real-time:

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d 
'{
    "text": "The system is under attack, we need to respond now!",
    "task_name": "Threat"
}'
```

## How to Contribute

This is a critical part of our roadmap, and we welcome contributions from the community. If you are interested in helping build out these features, please check out the issues under **Milestone 2** in our `PROJECT_ROADMAP.md` file on GitHub.
