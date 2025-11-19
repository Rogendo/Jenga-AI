
from datasets import load_dataset
from seq2seq_models.core.config import DataConfig

class DataProcessor:
    """
    A class for loading and preprocessing parallel data for Sequence-to-Sequence (Seq2Seq) models.

    This processor handles loading datasets, tokenizing source and target language texts,
    and preparing them for Seq2Seq model training.
    """
    def __init__(self, data_config: DataConfig, tokenizer):
        """
        Initializes the DataProcessor.

        Args:
            data_config (DataConfig): The configuration object defining the
                                      dataset to be processed.
            tokenizer: The tokenizer to use for processing the text data.
        """
        self.data_config = data_config
        self.tokenizer = tokenizer

    def create_dataset(self):
        """
        Loads, tokenizes, and prepares the parallel dataset based on the provided configuration.

        Returns:
            tuple: A tuple containing the training dataset and
                   the evaluation dataset (or None if no eval dataset).
        Raises:
            ValueError: If a dataset cannot be loaded or processed.
        """
        try:
            dataset = load_dataset(self.data_config.format, data_files=self.data_config.path)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from path '{self.data_config.path}' with format '{self.data_config.format}'. Check path and format. Error: {e}")

        prefix = "translate English to French: " # This prefix might need to be configurable or dynamic

        def preprocess_function(examples):
            """
            Preprocesses the examples by adding a prefix, tokenizing source and target languages.
            """
            inputs = [prefix + ex[self.data_config.source_lang] for ex in examples["translation"]]
            targets = [ex[self.data_config.target_lang] for ex in examples["translation"]]
            model_inputs = self.tokenizer(inputs, max_length=self.data_config.max_length, truncation=True)
            labels = self.tokenizer(text_target=targets, max_length=self.data_config.max_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        try:
            tokenized_datasets = dataset.map(preprocess_function, batched=True)
        except Exception as e:
            raise ValueError(f"Failed to tokenize dataset from path '{self.data_config.path}'. Ensure 'translation' column with '{self.data_config.source_lang}' and '{self.data_config.target_lang}' keys exists and tokenizer is compatible. Error: {e}")

        train_dataset = tokenized_datasets[self.data_config.train_split]
        eval_dataset = None
        if self.data_config.eval_split:
            eval_dataset = tokenized_datasets[self.data_config.eval_split]

        return train_dataset, eval_dataset
