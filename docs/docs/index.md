# Jenga-AI: A Unified Multi-Task Framework for African NLP

**Jenga-AI is an open-source framework designed to democratize Natural Language Processing for African languages and contexts.**

Our mission is to empower developers, researchers, and organizations across the continent to build, train, and deploy state-of-the-art NLP models that are efficient, powerful, and context-aware.

<div class="grid cards" markdown>
-   :material-rocket-launch:{ .lg .middle } **Train a Single Model for Multiple Tasks**

    ---

    Save significant computational resources and time by training one model to simultaneously perform classification, NER, question answering, and more.

-   :material-cogs:{ .lg .middle } **Simple, Config-Driven Experiments**

    ---

    Define and run complex training workflows with a single, easy-to-understand YAML file. No boilerplate code required.

-   :material-earth:{ .lg .middle } **Built for African Contexts**

    ---

    Move beyond generic Western models. Jenga-AI is designed to be fine-tuned on local languages and datasets, capturing the unique nuances of the African linguistic landscape.

</div>

## Get Started in Minutes

Ready to train your first model? Dive into our **[Quickstart Guide](getting-started/quickstart.md)** and see how easy it is to get started with Jenga-AI.

```python
# Conceptual Example: Train Sentiment & NER simultaneously
from multitask_bert import Trainer, ExperimentConfig

# 1. Load your experiment configuration from a YAML file
config = ExperimentConfig.from_yaml("experiment.yaml")

# 2. Initialize the Trainer
trainer = Trainer.from_config(config)

# 3. Train the model!
trainer.train()

# 4. (Coming Soon) Deploy the model with a single command
# trainer.deploy_as_api()
```

## Our Vision

We believe that the future of AI in Africa should be built *by* Africans, *for* Africans. Jenga-AI is more than a software library; it's a community-driven effort to build the tools and share the knowledge needed to solve our own challenges.

Whether you're tackling misinformation in Swahili, analyzing public policy documents, or building a customer service bot that understands Sheng, Jenga-AI provides the building blocks you need.
