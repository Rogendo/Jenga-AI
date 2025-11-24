# API Reference: Data Processing

The data processing components are responsible for loading raw data from your files, tokenizing it, and preparing it in the correct format for training with the `MultiTaskModel`.

## `DataProcessor`

**Class Path:** `multitask_bert.data.data_processing.DataProcessor`

The `DataProcessor` is the main class that handles all data preparation, driven by the `ExperimentConfig`.

### Role of the DataProcessor

1.  **Reads Configuration:** It inspects the `tasks` list in your `ExperimentConfig`.
2.  **Loads Data:** For each task, it loads the data file specified in `data_path`. It currently supports `.jsonl` files.
3.  **Processes and Tokenizes:** It applies task-specific processing logic to tokenize the text and format the labels correctly.
4.  **Automatic Label Discovery:** For NER and classification tasks, it can automatically discover the set of unique labels from your data and update the `ExperimentConfig` with the correct number of labels and a `label_map`.
5.  **Splits Data:** It splits the processed data into training and evaluation sets.
6.  **Formats for PyTorch:** It formats the datasets for efficient loading by a PyTorch `DataLoader`.

### Key Methods

- **`__init__(self, config, tokenizer)`**
    - **Purpose:** Initializes the `DataProcessor`.
    - **Arguments:**
        - `config (ExperimentConfig)`: The complete experiment configuration object.
        - `tokenizer`: The Hugging Face tokenizer instance to use for processing.

- **`process(self) -> Tuple[Dict, Dict, ExperimentConfig]`**
    - **Purpose:** The main method that orchestrates the entire data processing workflow.
    - **Process:**
        1.  Iterates through each `TaskConfig` in `config.tasks`.
        2.  Loads the corresponding data file into a Hugging Face `Dataset`.
        3.  Calls the appropriate internal processing method (e.g., `_process_ner`) based on the `task.type`.
        4.  Splits the resulting tokenized dataset into a training and evaluation set (currently an 80/20 split).
        5.  Collects all processed datasets into dictionaries.
    - **Returns:** A tuple containing:
        - `train_datasets (Dict[str, Dataset])`: A dictionary mapping task names to their training `Dataset` objects.
        - `eval_datasets (Dict[str, Dataset])`: A dictionary mapping task names to their evaluation `Dataset` objects.
        - `updated_config (ExperimentConfig)`: The experiment config, potentially updated with new label maps and counts discovered during processing.

## Expected Data Formats

The `DataProcessor` expects your `.jsonl` data files to follow specific formats depending on the task type.

### `single_label_classification`

Each line in the `.jsonl` file should be a JSON object with `text` and `label` keys.

```json
{"text": "This framework is wonderful and easy to use.", "label": "Positive"}
{"text": "I am very frustrated with the results.", "label": "Negative"}
```

### `multi_label_classification`

Each line should have a `text` key and a `labels` key containing a **list** of all applicable string labels.

```json
{"text": "The agent was proactive and solved the issue.", "labels": ["proactiveness", "resolution"]}
{"text": "The call started politely but the agent was not listening.", "labels": ["opening", "no_listening"]}
```

### `ner` (Named Entity Recognition)

Each line should have a `text` key and an `entities` key. `entities` should be a list of objects, where each object specifies the `label`, `start` character index, and `end` character index of an entity.

```json
{"text": "John Doe from Acme Corp is visiting Nairobi.", "entities": [{"label": "PERSON", "start": 0, "end": 8}, {"label": "ORG", "start": 14, "end": 23}, {"label": "LOCATION", "start": 37, "end": 44}]}
```
