# JengaAI: Multi-Task NLP Framework for Sustainable Development & Security

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Community](https://img.shields.io/badge/Join-Community-brightgreen)](https://github.com/JengaAI)

> **Tagline:** "Making Advanced NLP Accessible to Everyone, Everywhere"


JengaAI revolutionizes artificial intelligence accessibility across Kenya and Africa by introducing a groundbreaking multi-task learning framework that democratizes advanced NLP capabilities while understanding unique African contexts. Built on innovative mathematical fusion techniques, our solution enables a single model to simultaneously handle multiple critical tasks—from Swahili sentiment analysis and agricultural disease detection, cybersecurity threat identification to NLP in governance.

This framework directly addresses Kenya's national priorities through four transformative applications:

-   **National Security & Governance:** Unified threat detection (cyber attacks, hate speech, misinformation), multi-task document processing, and real-time Swahili-English security monitoring.
-   **Sustainable Economic Development:** Agricultural NLP (crop disease prediction, market analysis), financial inclusion (M-Pesa transaction analysis, fraud detection), and small business AI empowerment.
-   **Public Policy & Political Stability:** Multi-dimensional public sentiment analysis, policy document classification, and unified monitoring of development goals.
-   **Homegrown Generative AI:** African-context aware language models that understand local idioms, culturally relevant content generation, and Swahili-English code-switching comprehension.

##  Core Features

-   **Multi-Task Learning:** Train a single or multiple models on multiple NLP tasks saving time and resources.
-   **High Efficiency:** Achieve significant cost reduction and faster inference speeds.
-   **Extensible Task Library:** A growing collection of pre-built tasks for classification, NER, QA, and more.
-   **Fusion Mechanisms:** State-of-the-art techniques for combining and sharing information between tasks.
-   **African Context-Awareness:** Models designed to understand the nuances of African languages and contexts.

##  Technical Innovation

We pioneer attention fusion mechanisms and uncertainty-weighted multi-task learning—cutting-edge research techniques that automatically balance task importance while maintaining individual task expertise. Unlike imported Western models, JengaAI understands that "sukuma wiki" refers to both kale and economic resilience.

##  Project Architecture

The framework is organized as follows:

```
multitask_bert/
├── analysis/         # Attention visualization and model analysis
├── core/             # Core components: model, fusion, config
├── data/             # Data processing and loading pipelines
├── deployment/       # Inference and model exportation
├── tasks/            # Definitions for all NLP tasks
├── training/         # Trainer, callbacks, and data modules
└── utils/            # Utility functions
```

##  Getting Started

### Prerequisites

-   Python 3.9+
-   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/JengaAI.git
    cd JengaAI
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    *(Note: The `requirements.txt` file is currently empty. This will be populated with the necessary packages.)*
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

### Quickstart

Here is a conceptual example of how to use the framework to train a model on two different tasks:

```python
from multitask_bert.data import DataProcessor
from multitask_bert.tasks import ClassificationTask, NERTask
from multitask_bert.training import Trainer

# 1. Define your tasks
classification_task = ClassificationTask(
    name="SwahiliSentiment",
    label_map={0: "Negative", 1: "Positive"}
)
ner_task = NERTask(
    name="SecurityThreats",
    label_map={0: "O", 1: "B-Threat", 2: "I-Threat"}
)

# 2. Load and process your data
data_processor = DataProcessor()
train_data = data_processor.load_data(
    tasks=[classification_task, ner_task],
    train_files=["sentiment_data.csv", "threat_data.jsonl"]
)

# 3. Configure and run the trainer
trainer = Trainer(
    tasks=[classification_task, ner_task],
    model_name_or_path="bert-base-multilingual-cased",
    train_data=train_data
)

trainer.train()
```

##  Community & Contribution

JengaAI is an open-source project, and we welcome contributions from the community. Whether you're a developer, a researcher, or just passionate about our mission, there are many ways to get involved:

-   **Contribute Code:** Help us build new features, fix bugs, or improve the framework.
-   **Add a Task:** Introduce a new NLP task to the framework.
-   **Improve Documentation:** Help us make our documentation clearer and more comprehensive.
-   **Share Models:** Train and share models for different African languages and contexts.

Please read our `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

##  License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
