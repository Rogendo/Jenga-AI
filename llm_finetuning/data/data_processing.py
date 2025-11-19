
from datasets import load_dataset, concatenate_datasets
from llm_finetuning.core.config import DataConfig
from typing import List

class DataProcessor:
    """
    A class for loading and preprocessing data for Large Language Model (LLM) fine-tuning.

    This processor handles multiple datasets for multi-task learning and supports
    multi-lingual datasets, provided that the tokenizer used is also multi-lingual.
    """
    def __init__(self, data_configs: List[DataConfig], tokenizer):
        """
        Initializes the DataProcessor.

        Args:
            data_configs (List[DataConfig]): A list of configuration objects, each
                                              defining a dataset to be processed.
            tokenizer: The tokenizer to use for processing the text data.
        """
        self.data_configs = data_configs
        self.tokenizer = tokenizer

    def create_dataset(self):
        """
        Loads, tokenizes, and concatenates datasets based on the provided configurations.

        Returns:
            tuple: A tuple containing the concatenated training dataset and
                   the concatenated evaluation dataset (or None if no eval datasets).
        Raises:
            ValueError: If a dataset cannot be loaded or processed.
        """
        train_datasets = []
        eval_datasets = []

        for data_config in self.data_configs:
            try:
                dataset = load_dataset(data_config.format, data_files=data_config.path)
            except Exception as e:
                raise ValueError(f"Failed to load dataset from path '{data_config.path}' with format '{data_config.format}'. Check path and format. Error: {e}")
            
            def tokenize_function(examples):
                """
                Tokenizes the input text examples and creates labels for causal language modeling.
                """
                tokenized_inputs = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=data_config.max_length)
                tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
                return tokenized_inputs

            try:
                tokenized_datasets = dataset.map(tokenize_function, batched=True)
            except Exception as e:
                raise ValueError(f"Failed to tokenize dataset from path '{data_config.path}'. Ensure 'text' column exists and tokenizer is compatible. Error: {e}")
            
            train_datasets.append(tokenized_datasets[data_config.train_split])
            if data_config.eval_split:
                eval_datasets.append(tokenized_datasets[data_config.eval_split])

        train_dataset = concatenate_datasets(train_datasets)
        eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None

        return train_dataset, eval_dataset
