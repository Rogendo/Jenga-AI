
import torch # Import torch
import torch.nn as nn # Import torch.nn
from multitask_bert.tasks.base import BaseTask # Import BaseTask
from multitask_bert.tasks.classification import SingleLabelClassificationTask, MultiLabelClassificationTask # Import classification tasks
from multitask_bert.tasks.ner import NERTask # Import NERTask
from transformers import PretrainedConfig, AutoConfig # Import PretrainedConfig
import pytest
import yaml
from unittest.mock import patch, MagicMock
import os
from datasets import Dataset # Import Dataset
from torch.utils.data import DataLoader # Import DataLoader

from multitask_bert.core.config import ExperimentConfig, ModelConfig, TaskConfig, HeadConfig, TrainingConfig, LoggingConfig, TokenizerConfig
from multitask_bert.core.model import MultiTaskModel
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.training.trainer import Trainer

# Dummy data paths for testing
DUMMY_CLASSIFICATION_DATA_PATH = "multitask_bert/tests/dummy_classification_data.jsonl"
DUMMY_NER_DATA_PATH = "multitask_bert/tests/dummy_ner_data.jsonl"
DUMMY_QA_DATA_PATH = "multitask_bert/tests/dummy_qa_data.json"

@pytest.fixture
def multitask_bert_config_dict():
    return {
        "project_name": "test_project",
        "model": {
            "base_model": "distilbert-base-uncased",
        },
        "tokenizer": {
            "max_length": 128,
            "padding": "max_length",
            "truncation": True
        },
        "tasks": [
            {
                "name": "classification_task",
                "type": "classification",
                "data_path": DUMMY_CLASSIFICATION_DATA_PATH,
                "heads": [{"name": "cls_head", "num_labels": 3, "weight": 1.0}]
            },
            {
                "name": "ner_task",
                "type": "ner",
                "data_path": DUMMY_NER_DATA_PATH,
                "heads": [{"name": "ner_head", "num_labels": 5, "weight": 1.0}] # Example number of NER labels
            },
            {
                "name": "multi_label_classification_task",
                "type": "multi_label_classification",
                "data_path": DUMMY_CLASSIFICATION_DATA_PATH,
                "heads": [{"name": "mlc_head", "num_labels": 2, "weight": 1.0}]
            }
        ],
        "training": {
            "output_dir": "test_output_multitask_bert",
            "learning_rate": 5e-5,
            "batch_size": 8,
            "num_epochs": 1,
            "logging": {
                "service": "tensorboard",
                "experiment_name": "test_multitask_bert_run"
            }
        }
    }

@pytest.fixture
def multitask_bert_config(multitask_bert_config_dict):
    model_config = ModelConfig(**multitask_bert_config_dict['model'])
    tokenizer_config = TokenizerConfig(**multitask_bert_config_dict['tokenizer'])
    task_configs = []
    for task_dict in multitask_bert_config_dict['tasks']:
        heads = [HeadConfig(**head) for head in task_dict['heads']]
        task_configs.append(TaskConfig(
            name=task_dict['name'],
            type=task_dict['type'],
            data_path=task_dict['data_path'],
            heads=heads
        ))
    logging_config = LoggingConfig(**multitask_bert_config_dict['training']['logging'])
    training_config_dict = multitask_bert_config_dict['training'].copy()
    del training_config_dict['logging']
    training_config = TrainingConfig(**training_config_dict, logging=logging_config)
    return ExperimentConfig(
        project_name=multitask_bert_config_dict['project_name'],
        model=model_config,
        tokenizer=tokenizer_config,
        tasks=task_configs,
        training=training_config
    )

def test_multitask_bert_config_loading(multitask_bert_config_dict):
    # Test if config can be loaded and parsed correctly
    model_config = ModelConfig(**multitask_bert_config_dict['model'])
    tokenizer_config = TokenizerConfig(**multitask_bert_config_dict['tokenizer'])
    task_configs = []
    for task_dict in multitask_bert_config_dict['tasks']:
        heads = [HeadConfig(**head) for head in task_dict['heads']]
        task_configs.append(TaskConfig(
            name=task_dict['name'],
            type=task_dict['type'],
            data_path=task_dict['data_path'],
            heads=heads
        ))
    logging_config = LoggingConfig(**multitask_bert_config_dict['training']['logging'])
    training_config_dict = multitask_bert_config_dict['training'].copy()
    del training_config_dict['logging']
    training_config = TrainingConfig(**training_config_dict, logging=logging_config)
    
    config = ExperimentConfig(
        project_name=multitask_bert_config_dict['project_name'],
        model=model_config,
        tokenizer=tokenizer_config,
        tasks=task_configs,
        training=training_config
    )

    assert config.model.base_model == "distilbert-base-uncased"
    assert config.tokenizer.max_length == 128
    assert len(config.tasks) == 3
    assert config.tasks[0].name == "classification_task"
    assert config.tasks[1].data_path == DUMMY_NER_DATA_PATH
    assert config.training.output_dir == "test_output_multitask_bert"
    assert config.training.logging.service == "tensorboard"

# Mocking MultiTaskModel's __init__
@patch('multitask_bert.core.model.MultiTaskModel.__init__', return_value=None)
def test_multitask_bert_model_creation(mock_multitask_model_init, multitask_bert_config):
    mock_config_instance = MagicMock(spec=PretrainedConfig)
    mock_config_instance._attn_implementation = "eager" # Add this line

    # Create mock BaseTask instances
    mock_tasks = []
    for task_config in multitask_bert_config.tasks:
        mock_task = MagicMock(spec=BaseTask)
        mock_task.name = task_config.name
        mock_task.type = task_config.type
        mock_task.heads = MagicMock(spec=nn.ModuleDict) # Mock the heads attribute
        mock_tasks.append(mock_task)

    # Create a mock for the MultiTaskModel instance
    mock_model_instance = MagicMock()
    mock_model_instance.tasks = mock_tasks # Ensure the mock has a 'tasks' attribute

    # Call the constructor, passing the mock_model_instance as 'self'
    MultiTaskModel.__init__(mock_model_instance, mock_config_instance, multitask_bert_config.model, mock_tasks)

    mock_multitask_model_init.assert_called_once_with(mock_model_instance, mock_config_instance, multitask_bert_config.model, mock_tasks)
    assert mock_model_instance is not None
    assert len(mock_model_instance.tasks) == len(multitask_bert_config.tasks)

@patch('multitask_bert.core.model.AutoModel.from_pretrained')
@patch('multitask_bert.core.model.AutoConfig.from_pretrained')
def test_multitask_bert_forward(mock_auto_config, mock_auto_model, multitask_bert_config):
    # Mock the base model and config
    mock_encoder = MagicMock()
    mock_encoder.config.model_type = "bert" # Simulate a model that uses token_type_ids
    mock_encoder.return_value.last_hidden_state = torch.randn(2, 10, 768) # batch_size, seq_len, hidden_size
    mock_auto_model.return_value = mock_encoder
    mock_auto_config.return_value = MagicMock(spec=PretrainedConfig, hidden_size=768, _attn_implementation="eager")
    # Create mock BaseTask instances
    mock_task_output = MagicMock()
    mock_task_output.loss = torch.tensor(0.5)
    mock_task_output.logits = {"cls_head": torch.randn(2, 3)}

    mock_task1 = MagicMock(spec=BaseTask)
    mock_task1.get_forward_output.return_value = mock_task_output
    mock_task2 = MagicMock(spec=BaseTask)
    mock_task2.get_forward_output.return_value = mock_task_output
    mock_tasks = [mock_task1, mock_task2]

    # Create a MultiTaskModel instance
    model = MultiTaskModel(MagicMock(spec=PretrainedConfig, hidden_size=768, _attn_implementation="eager"), multitask_bert_config.model, mock_tasks)
    model.encoder = mock_encoder # Assign the mock encoder
    model.fusion = MagicMock() # Mock fusion layer
    model.fusion.return_value = torch.randn(2, 10, 768) # Output of fusion

    # Test case 1: with token_type_ids and fusion
    input_ids_1 = torch.randint(0, 100, (2, 10))
    attention_mask_1 = torch.ones(2, 10, dtype=torch.long)
    token_type_ids_1 = torch.zeros(2, 10, dtype=torch.long)
    task_id_1 = 0
    labels_1 = torch.randint(0, 3, (2,))

    # Re-mock encoder for this test case
    mock_encoder_case1 = MagicMock()
    mock_encoder_case1.config.model_type = "bert"
    mock_encoder_case1.return_value.last_hidden_state = torch.randn(2, 10, 768)
    model.encoder = mock_encoder_case1
    model.fusion = MagicMock() # Mock fusion layer
    model.fusion.return_value = torch.randn(2, 10, 768) # Output of fusion

    output = model.forward(input_ids_1, attention_mask_1, task_id_1, labels_1, token_type_ids=token_type_ids_1)

    mock_encoder_case1.assert_called_once_with(input_ids=input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)
    model.fusion.assert_called_once_with(mock_encoder_case1.return_value.last_hidden_state, task_id_1)
    mock_task1.get_forward_output.assert_called_once()
    assert output == mock_task_output

    # Reset mocks for next test case
    mock_task1.get_forward_output.reset_mock()
    mock_task2.get_forward_output.reset_called() # Use reset_called() instead of reset_mock() to keep return_value

    # Test case 2: without token_type_ids and without fusion
    input_ids_2 = torch.randint(0, 100, (2, 10))
    attention_mask_2 = torch.ones(2, 10, dtype=torch.long)
    task_id_2 = 1 # Use second task
    labels_2 = torch.randint(0, 3, (2,))

    # Re-mock encoder for this test case
    mock_encoder_case2 = MagicMock()
    mock_encoder_case2.config.model_type = "bert"
    mock_encoder_case2.return_value.last_hidden_state = torch.randn(2, 10, 768)
    model.encoder = mock_encoder_case2
    model.fusion = None # Disable fusion

    output = model.forward(input_ids_2, attention_mask_2, task_id_2, labels_2)

    mock_encoder_case2.assert_called_once_with(input_ids=input_ids_2, attention_mask=attention_mask_2, token_type_ids=None) # No token_type_ids
    # model.fusion.assert_not_called() # Fusion should not be called, but it's None
    mock_task2.get_forward_output.assert_called_once()
    assert output == mock_task_output

# Mocking DataProcessor for multitask_bert
@patch('multitask_bert.data.data_processing.DataProcessor.process')
def test_multitask_bert_data_processing(mock_data_processor_process, multitask_bert_config):
    mock_train_dataset = MagicMock()
    mock_eval_dataset = MagicMock()
    mock_data_processor_process.return_value = (
        {"classification_task": mock_train_dataset},
        {"ner_task": mock_eval_dataset},
        multitask_bert_config
    )

    mock_tokenizer = MagicMock()
    data_processor = DataProcessor(multitask_bert_config, mock_tokenizer)
    train_datasets, eval_datasets, _ = data_processor.process()

    mock_data_processor_process.assert_called_once()
    assert "classification_task" in train_datasets
    assert "ner_task" in eval_datasets
    assert train_datasets["classification_task"] is not None
    assert eval_datasets["ner_task"] is not None

def test_data_processor_tokenize(multitask_bert_config):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    data_processor = DataProcessor(multitask_bert_config, mock_tokenizer)
    
    batch = {"text": ["hello world", "test sentence"]}
    tokenized_batch = data_processor._tokenize(batch)

    mock_tokenizer.assert_called_once_with(
        batch['text'],
        padding=False,
        truncation=multitask_bert_config.tokenizer.truncation,
        max_length=multitask_bert_config.tokenizer.max_length,
    )
    assert tokenized_batch == {"input_ids": [1, 2], "attention_mask": [1, 1]}

def test_data_processor_process_single_label_classification(multitask_bert_config):
    mock_tokenizer = MagicMock()
    data_processor = DataProcessor(multitask_bert_config, mock_tokenizer)

    # Create a mock dataset for single-label classification
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.map.return_value = mock_dataset # map returns self for chaining
    mock_dataset.__len__.return_value = 1 # For iteration

    # Mock the example returned by the dataset iterator
    mock_example = {"text": "test", "label": 1}
    mock_dataset.__iter__.return_value = iter([mock_example])

    # Create a TaskConfig for single-label classification
    single_label_task_config = TaskConfig(
        name="single_label_task",
        type="single_label_classification",
        data_path="dummy.jsonl",
        heads=[HeadConfig(name="cls_head", num_labels=2, weight=1.0)]
    )

    processed_dataset = data_processor._process_single_label_classification(mock_dataset, single_label_task_config)

    # Assert that map was called
    mock_dataset.map.assert_called_once()
    # Assert that the mapped function correctly transformed the label
    # This is hard to assert directly on the mapped function, so we rely on the return value of map
    # and assume the internal logic of map is correct.
    # A more robust test would involve inspecting the call args of the mapped function.

def test_data_processor_process_multi_label_classification(multitask_bert_config):
    mock_tokenizer = MagicMock()
    data_processor = DataProcessor(multitask_bert_config, mock_tokenizer)

    # Create a mock dataset for multi-label classification
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.map.return_value = mock_dataset
    mock_dataset.__len__.return_value = 1

    mock_example = {"text": "test", "head1": [0.0, 1.0], "head2": [1.0, 0.0]}
    mock_dataset.__iter__.return_value = iter([mock_example])

    # Create a TaskConfig for multi-label classification
    multi_label_task_config = TaskConfig(
        name="multi_label_task",
        type="multi_label_classification",
        data_path="dummy.jsonl",
        heads=[
            HeadConfig(name="head1", num_labels=2, weight=1.0),
            HeadConfig(name="head2", num_labels=2, weight=1.0),
            HeadConfig(name="head3", num_labels=2, weight=1.0) # Missing in example
        ]
    )

    processed_dataset = data_processor._process_multi_label_classification(mock_dataset, multi_label_task_config)
    mock_dataset.map.assert_called_once()

def test_data_processor_process_ner(multitask_bert_config):
    mock_tokenizer = MagicMock()
    mock_tokenized_inputs = MagicMock()
    mock_tokenized_inputs.input_ids = [[1, 2, 3]]
    mock_tokenized_inputs.attention_mask = [[1, 1, 1]]
    mock_tokenized_inputs.offset_mapping = [[[0, 0], [0, 5], [6, 11]]] # CLS, "hello", "world"
    mock_tokenized_inputs.word_ids.return_value = [None, 0, 1] # Map tokens to words
    mock_tokenizer.return_value = mock_tokenized_inputs

    data_processor = DataProcessor(multitask_bert_config, mock_tokenizer)

    # Create a mock dataset for NER
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.__len__.return_value = 1

    mock_example = {
        "text": "hello world",
        "entities": [{"start": 0, "end": 5, "label": "LOC"}]
    }
    mock_dataset.__iter__.return_value = iter([mock_example])
    mock_dataset.column_names = ["text", "entities"] # Add column_names attribute

    # Create a TaskConfig for NER
    ner_task_config = TaskConfig(
        name="ner_task",
        type="ner",
        data_path="dummy.jsonl",
        heads=[HeadConfig(name="ner_head", num_labels=0, weight=1.0)] # num_labels will be updated
    )
    
    # Mock the map method to actually call the function passed to it
    def mock_map_side_effect(func, batched=False):
        if batched:
            # Simulate batched processing
            processed_examples = func({"text": [mock_example["text"]], "entities": [mock_example["entities"]]})
        else:
            processed_examples = func(mock_example)
        
        # Create a new mock dataset to return the processed examples
        processed_mock_dataset = MagicMock(spec=Dataset)
        processed_mock_dataset.__len__.return_value = 1
        processed_mock_dataset.__iter__.return_value = iter([processed_examples])
        processed_mock_dataset.column_names = list(processed_examples.keys())
        return processed_mock_dataset

    mock_dataset.map.side_effect = mock_map_side_effect

    processed_dataset = data_processor._process_ner(mock_dataset, ner_task_config)

    mock_dataset.map.assert_called_once()
    assert mock_tokenizer.call_count == 1 # Called once in align_labels_with_tokens
    assert ner_task_config.heads[0].num_labels > 0
    assert "ner_head" in ner_task_config.label_maps

def test_single_label_classification_task_init():
    config = TaskConfig(name="test_task", type="single_label_classification", data_path="dummy.jsonl", heads=[HeadConfig(name="head1", num_labels=3, weight=1.0)])
    task = SingleLabelClassificationTask(config)
    assert isinstance(task.heads["head1"], nn.Linear)
    assert task.heads["head1"].out_features == 3

def test_single_label_classification_task_forward_output():
    config = TaskConfig(name="test_task", type="single_label_classification", data_path="dummy.jsonl", heads=[HeadConfig(name="head1", num_labels=3, weight=1.0)])
    task = SingleLabelClassificationTask(config)
    
    pooled_output = torch.randn(2, 768) # batch_size, hidden_size
    labels = torch.tensor([0, 1], dtype=torch.long)
    feature = {"labels": labels}

    # Test with labels
    output = task.get_forward_output(feature, pooled_output)
    assert output.loss is not None
    assert "head1" in output.logits
    assert output.logits["head1"].shape == (2, 3)

    # Test without labels
    output_no_labels = task.get_forward_output({}, pooled_output)
    assert output_no_labels.loss == 0 # No loss calculated
    assert "head1" in output_no_labels.logits
    assert output_no_labels.logits["head1"].shape == (2, 3)

def test_multi_label_classification_task_init():
    config = TaskConfig(name="test_task", type="multi_label_classification", data_path="dummy.jsonl", heads=[HeadConfig(name="head1", num_labels=2, weight=1.0)])
    task = MultiLabelClassificationTask(config)
    assert isinstance(task.heads["head1"], nn.Linear)
    assert task.heads["head1"].out_features == 2

def test_multi_label_classification_task_forward_output():
    config = TaskConfig(name="test_task", type="multi_label_classification", data_path="dummy.jsonl", heads=[HeadConfig(name="head1", num_labels=2, weight=1.0)])
    task = MultiLabelClassificationTask(config)
    
    pooled_output = torch.randn(2, 768) # batch_size, hidden_size
    labels = {"head1": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float)}
    feature = {"labels": labels}

    # Test with labels
    output = task.get_forward_output(feature, pooled_output)
    assert output.loss is not None
    assert "head1" in output.logits
    assert output.logits["head1"].shape == (2, 2)

    # Test without labels
    output_no_labels = task.get_forward_output({}, pooled_output)
    assert output_no_labels.loss == 0 # No loss calculated
    assert "head1" in output_no_labels.logits
    assert output_no_labels.logits["head1"].shape == (2, 2)

def test_ner_task_init():
    config = TaskConfig(name="test_ner_task", type="ner", data_path="dummy.jsonl", heads=[HeadConfig(name="ner_head", num_labels=5, weight=1.0)])
    task = NERTask(config)
    assert isinstance(task.heads["ner_head"], nn.Linear)
    assert task.heads["ner_head"].out_features == 5

def test_ner_task_forward_output():
    config = TaskConfig(name="test_ner_task", type="ner", data_path="dummy.jsonl", heads=[HeadConfig(name="ner_head", num_labels=5, weight=1.0)])
    task = NERTask(config)
    
    sequence_output = torch.randn(2, 10, 768) # batch_size, seq_len, hidden_size
    labels = torch.randint(0, 5, (2, 10), dtype=torch.long)
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    feature_with_mask = {"labels": labels, "attention_mask": attention_mask}
    feature_no_mask = {"labels": labels}

    # Test with labels and attention_mask
    output_with_mask = task.get_forward_output(feature_with_mask, sequence_output)
    assert output_with_mask.loss is not None
    assert "ner_head" in output_with_mask.logits
    assert output_with_mask.logits["ner_head"].shape == (2, 10, 5)

    # Test with labels but no attention_mask
    output_no_mask = task.get_forward_output(feature_no_mask, sequence_output)
    assert output_no_mask.loss is not None
    assert "ner_head" in output_no_mask.logits
    assert output_no_mask.logits["ner_head"].shape == (2, 10, 5)

    # Test without labels
    output_no_labels = task.get_forward_output({}, sequence_output)
    assert output_no_labels.loss == 0 # No loss calculated
    assert "ner_head" in output_no_labels.logits
    assert output_no_labels.logits["ner_head"].shape == (2, 10, 5)

@patch('multitask_bert.training.trainer.SummaryWriter')
@patch('multitask_bert.training.trainer.mlflow')
def test_trainer_init_logger(mock_mlflow, mock_summary_writer, multitask_bert_config):
    # Test TensorBoard logger
    multitask_bert_config.training.logging.service = "tensorboard"
    trainer = Trainer(multitask_bert_config, MagicMock(), MagicMock(), {}, {})
    mock_summary_writer.assert_called_once()
    assert trainer.logger is not None
    trainer.close() # Close the logger

    # Test MLflow logger
    mock_summary_writer.reset_mock()
    mock_mlflow.set_experiment.reset_mock()
    mock_mlflow.start_run.reset_mock()
    multitask_bert_config.training.logging.service = "mlflow"
    trainer = Trainer(multitask_bert_config, MagicMock(), MagicMock(), {}, {})
    mock_mlflow.set_experiment.assert_called_once()
    mock_mlflow.start_run.assert_called_once()
    assert trainer.logger is not None
    trainer.close() # Close the logger

def test_trainer_create_dataloaders(multitask_bert_config):
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    
    mock_dataset_cls = MagicMock(spec=Dataset)
    mock_dataset_cls.__len__.return_value = 10
    mock_dataset_cls.column_names = ["input_ids", "attention_mask", "labels"]
    mock_dataset_cls.__getitem__.return_value = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor(0)
    }

    mock_dataset_ner = MagicMock(spec=Dataset)
    mock_dataset_ner.__len__.return_value = 10
    mock_dataset_ner.column_names = ["input_ids", "attention_mask", "labels"]
    mock_dataset_ner.__getitem__.return_value = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor([0, 1, 2])
    }

    multitask_bert_config.tasks[0].type = "single_label_classification"
    multitask_bert_config.tasks[1].type = "ner"

    trainer = Trainer(multitask_bert_config, MagicMock(), mock_tokenizer, 
                      {"classification_task": mock_dataset_cls}, 
                      {"ner_task": mock_dataset_ner})
    
    assert "classification_task" in trainer.train_dataloaders
    assert "ner_task" in trainer.eval_dataloaders
    assert isinstance(trainer.train_dataloaders["classification_task"], DataLoader)
    assert isinstance(trainer.eval_dataloaders["ner_task"], DataLoader)

@patch('multitask_bert.training.trainer.AdamW')
@patch('multitask_bert.training.trainer.get_linear_schedule_with_warmup')
def test_trainer_create_optimizer_and_scheduler(mock_get_linear_schedule_with_warmup, mock_adamw, multitask_bert_config):
    mock_model = MagicMock()
    mock_model.parameters.return_value = []
    trainer = Trainer(multitask_bert_config, mock_model, MagicMock(), {}, {})
    
    num_training_steps = 100
    optimizer, scheduler = trainer._create_optimizer_and_scheduler(num_training_steps)
    
    mock_adamw.assert_called_once()
    mock_get_linear_schedule_with_warmup.assert_called_once()
    assert optimizer is not None
    assert scheduler is not None

@patch('multitask_bert.training.trainer.DataLoader')
@patch('multitask_bert.training.trainer.Trainer._log_metrics')
@patch('multitask_bert.training.trainer.Trainer.evaluate')
@patch('multitask_bert.training.trainer.tqdm')
def test_trainer_train(mock_tqdm, mock_evaluate, mock_log_metrics, mock_dataloader_class, multitask_bert_config):
    mock_model = MagicMock()
    mock_model.train.return_value = None
    mock_model.to.return_value = mock_model
    mock_model.return_value.loss = torch.tensor(0.1, requires_grad=True)
    mock_model.return_value.logits = {"cls_head": torch.randn(8, 3)}

    mock_optimizer = MagicMock()
    mock_scheduler = MagicMock()

    # Configure the mocked DataLoader class
    mock_dataloader_instance = MagicMock(spec=DataLoader)
    mock_dataloader_instance.__len__.return_value = 1
    mock_dataloader_instance.__iter__.return_value = iter([
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": torch.tensor([0, 1])
        }
    ])
    mock_dataloader_class.return_value = mock_dataloader_instance # When DataLoader is called, return our mock instance

    multitask_bert_config.training.num_epochs = 1
    multitask_bert_config.training.early_stopping_patience = None # Disable early stopping for this test

    trainer = Trainer(multitask_bert_config, mock_model, MagicMock(), 
                      {"classification_task": MagicMock(spec=Dataset)}, # Pass a mock dataset, but it won't be used directly
                      {"classification_task": MagicMock(spec=Dataset)}) # Pass a mock dataset, but it won't be used directly
    
    trainer._create_optimizer_and_scheduler = MagicMock(return_value=(mock_optimizer, mock_scheduler))
    mock_evaluate.return_value = {"eval_loss": 0.5}

    trainer.train()

    mock_model.train.assert_called()
    mock_optimizer.zero_grad.assert_called()
    mock_model.assert_called() # model should be called in the training loop
    mock_optimizer.step.assert_called()
    mock_scheduler.step.assert_called()
    mock_log_metrics.assert_called() # Called for train loss and eval metrics
    mock_evaluate.assert_called_once()

@patch('multitask_bert.training.trainer.Trainer._create_dataloaders')
@patch('multitask_bert.training.trainer.compute_classification_metrics')
@patch('multitask_bert.training.trainer.compute_multi_label_metrics')
@patch('multitask_bert.training.trainer.compute_ner_metrics')
@patch('multitask_bert.training.trainer.tqdm')
def test_trainer_evaluate(mock_tqdm, mock_compute_ner_metrics, mock_compute_multi_label_metrics, mock_compute_classification_metrics, mock_create_dataloaders, multitask_bert_config):
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    
    # Configure mock_model to return an object with a logits attribute
    mock_outputs = MagicMock()
    mock_outputs.logits = {
        "cls_head": torch.tensor([[0.1, 0.8, 0.1]]),
        "mlc_head": torch.tensor([[0.2, 0.7]]),
        "ner_head": torch.tensor([[[0.9, 0.05, 0.05, 0.0, 0.0]]])
    }
    mock_model.return_value = mock_outputs # When mock_model is called, return mock_outputs

    # Configure mock_model to return an object with a logits attribute
    mock_outputs = MagicMock()
    mock_outputs.logits = {
        "cls_head": torch.tensor([[0.1, 0.8, 0.1]]),
        "mlc_head": torch.tensor([[0.2, 0.7]]),
        "ner_head": torch.tensor([[[0.9, 0.05, 0.05, 0.0, 0.0]]])
    }
    mock_model.return_value = mock_outputs # When mock_model is called, return mock_outputs

    mock_dataloader_cls_batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([0])
        }
    ]
    mock_dataloader_cls = MagicMock(spec=DataLoader)
    mock_dataloader_cls.__len__.return_value = 1
    mock_dataloader_cls.__iter__.return_value = iter(mock_dataloader_cls_batches)

    mock_dataloader_mlc_batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": {"mlc_head": torch.tensor([[0.0, 1.0]])}
        }
    ]
    mock_dataloader_mlc = MagicMock(spec=DataLoader)
    mock_dataloader_mlc.__len__.return_value = 1
    mock_dataloader_mlc.__iter__.return_value = iter(mock_dataloader_mlc_batches)

    mock_dataloader_ner_batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[0, 1, 2]])
        }
    ]
    mock_dataloader_ner = MagicMock(spec=DataLoader)
    mock_dataloader_ner.__len__.return_value = 1
    mock_dataloader_ner.__iter__.return_value = iter(mock_dataloader_ner_batches)

    mock_create_dataloaders.return_value = {
        "classification_task": mock_dataloader_cls, 
        "multi_label_classification_task": mock_dataloader_mlc,
        "ner_task": mock_dataloader_ner
    }

    # Configure mock_tqdm to yield the batches from the dataloaders
    def tqdm_side_effect(iterable, desc):
        return iterable # Just return the iterable directly

    mock_tqdm.side_effect = tqdm_side_effect

    multitask_bert_config.tasks[0].type = "single_label_classification"
    multitask_bert_config.tasks[1].type = "multi_label_classification"
    multitask_bert_config.tasks[2].type = "ner"
    multitask_bert_config.tasks[2].label_maps = {"ner_head": {0: "O", 1: "LOC", 2: "PER", 3: "ORG", 4: "MISC"}}

    mock_compute_classification_metrics.return_value = {"accuracy": 0.9}
    mock_compute_multi_label_metrics.return_value = {"f1": 0.8}
    mock_compute_ner_metrics.return_value = {"f1": 0.7}

    trainer = Trainer(multitask_bert_config, mock_model, MagicMock(), {}, {})
    
    metrics = trainer.evaluate()

    mock_model.eval.assert_called_once()
    mock_model.assert_called() # model should be called in the evaluation loop
    mock_compute_classification_metrics.assert_called_once()
    mock_compute_multi_label_metrics.assert_called_once()
    mock_compute_ner_metrics.assert_called_once()
    assert "eval_overall_f1" in metrics
    assert "eval_loss" in metrics

@patch('multitask_bert.training.trainer.SummaryWriter')
@patch('multitask_bert.training.trainer.mlflow')
def test_trainer_log_metrics(mock_mlflow, mock_summary_writer, multitask_bert_config):
    metrics = {"loss": 0.1, "accuracy": 0.9}
    step = 10

    # Test TensorBoard logging
    multitask_bert_config.training.logging.service = "tensorboard"
    trainer = Trainer(multitask_bert_config, MagicMock(), MagicMock(), {}, {})
    trainer.logger = mock_summary_writer.return_value # Assign mock logger
    trainer._log_metrics(metrics, step, "Train")
    mock_summary_writer.return_value.add_scalar.assert_called()

    # Test MLflow logging
    mock_summary_writer.reset_mock()
    mock_mlflow.log_metrics.reset_mock()
    multitask_bert_config.training.logging.service = "mlflow"
    trainer = Trainer(multitask_bert_config, MagicMock(), MagicMock(), {}, {})
    trainer.logger = mock_mlflow # Assign mock logger
    trainer._log_metrics(metrics, step, "Eval")
    mock_mlflow.log_metrics.assert_called_once_with(metrics, step=step)

@patch('multitask_bert.training.trainer.SummaryWriter')
@patch('multitask_bert.training.trainer.mlflow')
def test_trainer_close(mock_mlflow, mock_summary_writer, multitask_bert_config):
    # Test TensorBoard close
    multitask_bert_config.training.logging.service = "tensorboard"
    trainer = Trainer(multitask_bert_config, MagicMock(), MagicMock(), {}, {})
    trainer.logger = mock_summary_writer.return_value
    trainer.close()
    mock_summary_writer.return_value.close.assert_called_once()

    # Test MLflow close
    mock_summary_writer.reset_mock()
    mock_mlflow.end_run.reset_mock()
    mock_mlflow.start_run.return_value = MagicMock() # Mock start_run to return an active run
    
    multitask_bert_config.training.logging.service = "mlflow"
    trainer = Trainer(multitask_bert_config, MagicMock(), MagicMock(), {}, {})
    trainer.logger = mock_mlflow # trainer.logger is now the mocked mlflow module
    trainer.close()
    mock_mlflow.end_run.assert_called_once()
