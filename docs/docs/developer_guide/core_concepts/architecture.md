# Framework Architecture

The Jenga-NLP framework is designed with modularity and extensibility in mind. It follows a config-driven approach, allowing you to define and manage complex multi-task learning experiments with ease.

## Core Principles

- **Modularity:** The framework is divided into distinct components, each responsible for a specific part of the NLP pipeline. This makes it easy to understand, maintain, and extend the codebase.
- **Extensibility:** Adding new tasks, models, or datasets is a straightforward process that involves implementing a few base classes and updating the configuration.
- **Config-Driven:** All aspects of an experiment, from the model architecture to the training process, are defined in a single YAML configuration file. This ensures reproducibility and simplifies experiment management.

## High-Level Overview

The framework can be broken down into the following key components:

1.  **Configuration (`multitask_bert/core/config.py`):** This component is responsible for parsing the `experiment.yaml` file and creating the necessary configuration objects. It uses Python's `dataclasses` to define a clear and type-safe configuration structure.

2.  **Data Processing (`multitask_bert/data/data_processing.py`):** The `DataProcessor` class handles the loading and preprocessing of data for all tasks. It takes the raw data paths from the configuration and converts them into tokenized datasets ready for training.

3.  **Tasks (`multitask_bert/tasks/`):** Each task (e.g., classification, NER) is represented by a `BaseTask` subclass. These classes define the task-specific heads, loss functions, and forward pass logic.

4.  **Model (`multitask_bert/core/model.py`):** The `MultiTaskModel` is the heart of the framework. It consists of a shared encoder (e.g., `distilbert-base-uncased`) and a set of task-specific heads. It can also include an optional `AttentionFusion` layer to learn task-specific representations.

5.  **Trainer (`multitask_bert/training/trainer.py`):** The `Trainer` class orchestrates the entire training and evaluation process. It uses a round-robin approach to sample batches from different tasks and handles optimization, scheduling, and metric calculation.

6.  **Logging (`multitask_bert/training/trainer.py`):** The framework is integrated with MLflow and TensorBoard for experiment tracking. The `Trainer` class initializes the logger and logs metrics during training and evaluation.

7.  **Deployment (`multitask_bert/deployment/`):** This component provides tools for exporting trained models for inference and deploying them in production environments.

## Data Flow

The following diagram illustrates the data flow within the framework:

```
[experiment.yaml] -> [DataProcessor] -> [Tokenized Datasets]
                                              |
                                              v
+-------------------------------------------------------------------------+
|                                  Trainer                                  |
|                                                                         |
|  +-----------------+      +-----------------+      +-----------------+  |
|  | Task 1 Dataloader|      | Task 2 Dataloader|      | Task 3 Dataloader|  |
|  +-----------------+      +-----------------+      +-----------------+  |
|          |                      |                      |                 |
|          +----------------------+----------------------+                 |
|                                 |                                        |
|                                 v                                        |
|  +---------------------------------------------------------------------+  |
|  |                             MultiTaskModel                            |  |
|  |                                                                     |  |
|  |  +-----------------------------------------------------------------+  |  |
|  |  |                         Shared Encoder                          |  |  |
|  |  +-----------------------------------------------------------------+  |  |
|  |                                 |                                     |  |
|  |  +-----------------------------------------------------------------+  |  |
|  |  |                       (Optional) Fusion Layer                     |  |  |
|  |  +-----------------------------------------------------------------+  |  |
|  |                                 |                                     |  |
|  |  +----------+      +----------+      +----------+                     |  |
|  |  | Task 1 Head|      | Task 2 Head|      | Task 3 Head|                     |  |
|  |  +----------+      +----------+      +----------+                     |  |
|  +---------------------------------------------------------------------+  |
|                                 |                                        |
|                                 v                                        |
|  +---------------------------------------------------------------------+  |
|  |                                Metrics                                |  |
|  +---------------------------------------------------------------------+  |
|                                 |                                        |
|                                 v                                        |
|  +---------------------------------------------------------------------+  |
|  |                                Logger                                 |  |
|  |                        (MLflow / TensorBoard)                         |  |
|  +---------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```
