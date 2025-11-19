import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from typing import Dict, Tuple, List
from ..core.config import ExperimentConfig, TaskConfig
from datasets import Dataset
import json # Added import

class DataProcessor:
    """
    Processes raw data from files into tokenized features for various tasks,
    driven by a unified ExperimentConfig.
    """
    def __init__(self, config: ExperimentConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def process(self) -> Tuple[Dict[str, Dataset], Dict[str, Dataset], ExperimentConfig]:
        """
        Main processing method. Iterates through tasks, processes data,
        and returns dictionaries of train and eval Datasets.
        """
        train_datasets = {}
        eval_datasets = {}

        for task_config in self.config.tasks:
            print(f"Processing data for task: {task_config.name}")
            
            # Custom loading for NER to debug pandas issue
            if task_config.type == "ner":
                data_list = []
                with open(task_config.data_path, 'r') as f:
                    for line in f:
                        data_list.append(json.loads(line))
                df = pd.DataFrame(data_list)
            else:
                df = pd.read_json(task_config.data_path, lines=task_config.data_path.endswith('.jsonl'))
            
            dataset = Dataset.from_pandas(df)
            
            if task_config.type == "single_label_classification":
                dataset = self._process_single_label_classification(dataset, task_config)
            elif task_config.type == "multi_label_classification":
                dataset = self._process_multi_label_classification(dataset, task_config)
            elif task_config.type == "ner":
                dataset = self._process_ner(dataset, task_config)

            # Tokenize
            tokenized_dataset = dataset.map(self._tokenize, batched=True)
            
            # Set format
            columns_to_set = ['input_ids', 'attention_mask']
            if 'labels' in tokenized_dataset.column_names:
                columns_to_set.append('labels')
            tokenized_dataset.set_format(type='torch', columns=columns_to_set)

            # For this example, we'll just split the data.
            # A proper implementation would use pre-defined splits.
            split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
            train_datasets[task_config.name] = split_dataset['train']
            eval_datasets[task_config.name] = split_dataset['test']
        
        return train_datasets, eval_datasets, self.config

    def _tokenize(self, batch):
        """Generic tokenization function."""
        return self.tokenizer(
            batch['text'],
            padding=False, # Padding will be handled by the DataLoader's collate_fn
            truncation=self.config.tokenizer.truncation,
            max_length=self.config.tokenizer.max_length,
        )

    def _process_single_label_classification(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Processes data for single-label classification tasks."""
        # Assumes 'text' and 'label' columns in the dataset
        # The label is directly used as a long tensor
        def map_labels(example):
            example['labels'] = torch.tensor(example['label'], dtype=torch.long)
            return example
        return dataset.map(map_labels)

    def _process_multi_label_classification(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Processes data for multi-label classification tasks."""
        # Assumes 'text' and a dictionary of labels matching head names
        def map_labels(example):
            labels_dict = {}
            for head_config in task_config.heads:
                # This assumes the label structure matches the head names
                if head_config.name in example: # Check if the head name exists as a column
                    labels_dict[head_config.name] = torch.tensor(example[head_config.name], dtype=torch.float)
                else:
                    # If not found, create a dummy tensor (e.g., for inference or missing data)
                    labels_dict[head_config.name] = torch.zeros(head_config.num_labels, dtype=torch.float)
            example['labels'] = labels_dict
            return example
        return dataset.map(map_labels)

    def _process_ner(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Processes data for NER tasks, including complex label alignment."""
        
        unique_labels = set(['O'])
        for example in dataset:
            for entity in example['entities']:
                unique_labels.add(entity['label'])
        
        label_to_id = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
        id_to_label = {i: label for label, i in label_to_id.items()}
        
        task_config.heads[0].num_labels = len(label_to_id)
        task_config.label_maps = {'ner_head': id_to_label}

        def align_labels_with_tokens(examples):
            tokenized_inputs = self.tokenizer(
                examples['text'],
                padding=self.config.tokenizer.padding,
                truncation=self.config.tokenizer.truncation,
                max_length=self.config.tokenizer.max_length,
                is_split_into_words=False,
                return_offsets_mapping=True # We need offset mapping to find character spans
            )

            labels = []
            for batch_index in range(len(examples['text'])):
                word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
                offset_mapping = tokenized_inputs['offset_mapping'][batch_index]
                
                # Initialize labels for this example
                current_labels = [-100] * len(word_ids)
                
                # Map entities to token labels
                for entity in examples['entities'][batch_index]:
                    start_char = entity['start']
                    end_char = entity['end']
                    entity_label_id = label_to_id[entity['label']]
                    
                    # Find the tokens that cover this entity
                    token_start_index = -1
                    token_end_index = -1
                    
                    for i, (offset_start, offset_end) in enumerate(offset_mapping):
                        if offset_start == 0 and offset_end == 0: # Special tokens like [CLS], [SEP]
                            continue
                        
                        # Check for overlap
                        if max(start_char, offset_start) < min(end_char, offset_end):
                            if token_start_index == -1:
                                token_start_index = i
                            token_end_index = i
                    
                    if token_start_index != -1 and token_end_index != -1:
                        # Assign B-I-O labels
                        current_labels[token_start_index] = entity_label_id # B-TAG
                        for i in range(token_start_index + 1, token_end_index + 1):
                            current_labels[i] = entity_label_id # I-TAG (for simplicity, using same ID)
                            
                labels.append(current_labels)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        return dataset.map(align_labels_with_tokens, batched=True)