
import pytest
import yaml
from unittest.mock import patch, MagicMock
import os

from llm_finetuning.core.config import LLMFinetuningConfig, ModelConfig, DataConfig, TrainingConfig, PeftConfig, TeacherStudentConfig, LoggingConfig
from llm_finetuning.pipeline import FinetuningPipeline
from llm_finetuning.model.model_factory import ModelFactory
from llm_finetuning.data.data_processing import DataProcessor
from llm_finetuning.training.trainer import Trainer

# Dummy data paths for testing
DUMMY_LLM_DATA_PATH = "tests/dummy_llm_data.json"

@pytest.fixture
def llm_config_dict():
    return {
        "model": {
            "name": "EleutherAI/gpt-neo-125M",
            "quantization": None,
            "peft_config": {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"]
            },
            "teacher_student_config": None
        },
        "data": [
            {
                "path": DUMMY_LLM_DATA_PATH,
                "format": "json",
                "train_split": "train",
                "eval_split": "validation",
                "max_length": 512
            }
        ],
        "training": {
            "output_dir": "test_output_llm",
            "learning_rate": 2e-4,
            "batch_size": 1,
            "num_epochs": 1,
            "gradient_accumulation_steps": 1,
            "logging_steps": 10,
            "save_steps": 100,
            "logging_config": {
                "report_to": "tensorboard",
                "run_name": "test_llm_run"
            }
        }
    }

@pytest.fixture
def llm_config(llm_config_dict):
    peft_config = PeftConfig(**llm_config_dict['model']['peft_config']) if llm_config_dict['model'].get('peft_config') else None
    teacher_student_config = TeacherStudentConfig(**llm_config_dict['model']['teacher_student_config']) if llm_config_dict['model'].get('teacher_student_config') else None
    model_config = ModelConfig(
        name=llm_config_dict['model']['name'],
        quantization=llm_config_dict['model'].get('quantization'),
        peft_config=peft_config,
        teacher_student_config=teacher_student_config
    )
    data_configs = [DataConfig(**data_config) for data_config in llm_config_dict['data']]
    logging_config = LoggingConfig(**llm_config_dict['training']['logging_config']) if llm_config_dict['training'].get('logging_config') else None
    training_config = TrainingConfig(
        output_dir=llm_config_dict['training']['output_dir'],
        learning_rate=llm_config_dict['training']['learning_rate'],
        batch_size=llm_config_dict['training']['batch_size'],
        num_epochs=llm_config_dict['training']['num_epochs'],
        gradient_accumulation_steps=llm_config_dict['training']['gradient_accumulation_steps'],
        logging_steps=llm_config_dict['training']['logging_steps'],
        save_steps=llm_config_dict['training']['save_steps'],
        logging_config=logging_config
    )
    return LLMFinetuningConfig(model=model_config, data=data_configs, training=training_config)

def test_llm_config_loading(llm_config_dict):
    # Test if config can be loaded and parsed correctly
    peft_config = PeftConfig(**llm_config_dict['model']['peft_config'])
    model_config = ModelConfig(name=llm_config_dict['model']['name'], peft_config=peft_config)
    data_configs = [DataConfig(**data_config) for data_config in llm_config_dict['data']]
    logging_config = LoggingConfig(**llm_config_dict['training']['logging_config'])
    training_config_dict = llm_config_dict['training'].copy()
    del training_config_dict['logging_config'] # Remove to avoid TypeError
    training_config = TrainingConfig(**training_config_dict, logging_config=logging_config)
    
    config = LLMFinetuningConfig(model=model_config, data=data_configs, training=training_config)

    assert config.model.name == "EleutherAI/gpt-neo-125M"
    assert config.data[0].path == DUMMY_LLM_DATA_PATH
    assert config.training.output_dir == "test_output_llm"
    assert config.training.logging_config.report_to == "tensorboard"

@patch('llm_finetuning.model.model_factory.AutoModelForCausalLM.from_pretrained')
@patch('llm_finetuning.model.model_factory.AutoTokenizer.from_pretrained')
@patch('llm_finetuning.model.model_factory.get_peft_model')
def test_llm_model_creation(mock_get_peft_model, mock_auto_tokenizer, mock_auto_model, llm_config):
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.return_value = mock_tokenizer_instance
    mock_model_instance = MagicMock()
    mock_auto_model.return_value = mock_model_instance
    mock_peft_model_instance = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model_instance

    model_factory = ModelFactory(llm_config.model)
    model, tokenizer = model_factory.create_model()

    mock_auto_model.assert_called_once_with(llm_config.model.name, quantization_config=None, device_map="auto")
    mock_auto_tokenizer.assert_called_once_with(llm_config.model.name)
    mock_get_peft_model.assert_called_once()
    assert model == mock_peft_model_instance
    assert tokenizer == mock_tokenizer_instance

@patch('llm_finetuning.data.data_processing.load_dataset')
@patch('llm_finetuning.data.data_processing.concatenate_datasets')
def test_llm_data_processing(mock_concatenate_datasets, mock_load_dataset, llm_config):
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.map.return_value = {"train": MagicMock(), "validation": MagicMock()}
    mock_load_dataset.return_value = mock_dataset_instance

    mock_tokenizer = MagicMock()
    data_processor = DataProcessor(llm_config.data, mock_tokenizer)
    train_dataset, eval_dataset = data_processor.create_dataset()

    mock_load_dataset.assert_called_once_with(llm_config.data[0].format, data_files=llm_config.data[0].path)
    mock_concatenate_datasets.assert_called() # Called for train and potentially eval
    assert train_dataset is not None
    assert eval_dataset is not None

@patch('llm_finetuning.training.base_trainer.HuggingFaceTrainer')
@patch('llm_finetuning.training.base_trainer.TrainingArguments')
def test_llm_trainer_initialization(mock_training_arguments, mock_huggingface_trainer, llm_config):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_train_dataset = MagicMock()
    mock_eval_dataset = MagicMock()

    trainer_instance = Trainer(mock_model, mock_tokenizer, mock_train_dataset, mock_eval_dataset, llm_config.training)
    trainer_instance.train()

    mock_training_arguments.assert_called_once()
    mock_huggingface_trainer.assert_called_once_with(
        model=mock_model,
        args=mock_training_arguments.return_value,
        train_dataset=mock_train_dataset,
        eval_dataset=mock_eval_dataset,
        tokenizer=mock_tokenizer,
    )
    mock_huggingface_trainer.return_value.train.assert_called_once()

