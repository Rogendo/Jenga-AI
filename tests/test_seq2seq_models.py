
import pytest
import yaml
from unittest.mock import patch, MagicMock
import os

from seq2seq_models.core.config import Seq2SeqFinetuningConfig, ModelConfig, DataConfig, TrainingConfig, LoggingConfig, TeacherStudentConfig
from seq2seq_models.model.seq2seq_model import Seq2SeqModel
from seq2seq_models.data.data_processing import DataProcessor
from seq2seq_models.training.trainer import Trainer

# Dummy data paths for testing
DUMMY_SEQ2SEQ_DATA_PATH = "tests/dummy_seq2seq_data.json"

@pytest.fixture
def seq2seq_config_dict():
    return {
        "model": {
            "name": "Helsinki-NLP/opus-mt-en-fr",
            "teacher_student_config": {
                "teacher_model": "Helsinki-NLP/opus-mt-en-fr",
                "distillation_alpha": 0.5
            }
        },
        "data": {
            "path": DUMMY_SEQ2SEQ_DATA_PATH,
            "format": "json",
            "source_lang": "en",
            "target_lang": "fr",
            "train_split": "train",
            "eval_split": "validation",
            "max_length": 128
        },
        "training": {
            "output_dir": "test_output_seq2seq",
            "learning_rate": 2e-5,
            "batch_size": 4,
            "num_epochs": 3,
            "gradient_accumulation_steps": 1,
            "logging_steps": 10,
            "save_steps": 100,
            "logging_config": {
                "report_to": "tensorboard",
                "run_name": "test_seq2seq_run"
            }
        }
    }

@pytest.fixture
def seq2seq_config(seq2seq_config_dict):
    teacher_student_config = TeacherStudentConfig(**seq2seq_config_dict['model']['teacher_student_config']) if seq2seq_config_dict['model'].get('teacher_student_config') else None
    model_config = ModelConfig(
        name=seq2seq_config_dict['model']['name'],
        teacher_student_config=teacher_student_config
    )
    data_config = DataConfig(**seq2seq_config_dict['data'])
    logging_config = LoggingConfig(**seq2seq_config_dict['training']['logging_config']) if seq2seq_config_dict['training'].get('logging_config') else None
    training_config = TrainingConfig(
        output_dir=seq2seq_config_dict['training']['output_dir'],
        learning_rate=seq2seq_config_dict['training']['learning_rate'],
        batch_size=seq2seq_config_dict['training']['batch_size'],
        num_epochs=seq2seq_config_dict['training']['num_epochs'],
        gradient_accumulation_steps=seq2seq_config_dict['training']['gradient_accumulation_steps'],
        logging_steps=seq2seq_config_dict['training']['logging_steps'],
        save_steps=seq2seq_config_dict['training']['save_steps'],
        logging_config=logging_config
    )
    return Seq2SeqFinetuningConfig(model=model_config, data=data_config, training=training_config)

def test_seq2seq_config_loading(seq2seq_config_dict):
    # Test if config can be loaded and parsed correctly
    teacher_student_config = TeacherStudentConfig(**seq2seq_config_dict['model']['teacher_student_config'])
    model_config = ModelConfig(name=seq2seq_config_dict['model']['name'], teacher_student_config=teacher_student_config)
    data_config = DataConfig(**seq2seq_config_dict['data'])
    logging_config = LoggingConfig(**seq2seq_config_dict['training']['logging_config'])
    training_config_dict = seq2seq_config_dict['training'].copy()
    del training_config_dict['logging_config'] # Remove to avoid TypeError
    training_config = TrainingConfig(**training_config_dict, logging_config=logging_config)
    
    config = Seq2SeqFinetuningConfig(model=model_config, data=data_config, training=training_config)

    assert config.model.name == "Helsinki-NLP/opus-mt-en-fr"
    assert config.data.path == DUMMY_SEQ2SEQ_DATA_PATH
    assert config.training.output_dir == "test_output_seq2seq"
    assert config.training.logging_config.report_to == "tensorboard"

@patch('seq2seq_models.model.seq2seq_model.AutoModelForSeq2SeqLM.from_pretrained')
@patch('seq2seq_models.model.seq2seq_model.AutoTokenizer.from_pretrained')
def test_seq2seq_model_creation(mock_auto_tokenizer, mock_auto_model, seq2seq_config):
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.return_value = mock_tokenizer_instance
    mock_model_instance = MagicMock()
    mock_auto_model.return_value = mock_model_instance

    seq2seq_model_instance = Seq2SeqModel(seq2seq_config.model)
    model, tokenizer = seq2seq_model_instance.create_model_and_tokenizer()

    assert mock_auto_model.call_count == 2 # Called once for student, once for teacher
    mock_auto_tokenizer.assert_called_once_with(seq2seq_config.model.name)
    assert model is not None # Will be wrapped by TeacherStudentModel
    assert tokenizer == mock_tokenizer_instance

@patch('seq2seq_models.data.data_processing.load_dataset')
def test_seq2seq_data_processing(mock_load_dataset, seq2seq_config):
    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = MagicMock(map=MagicMock(return_value={"train": MagicMock(), "validation": MagicMock()}))
    mock_load_dataset.return_value = mock_dataset

    mock_tokenizer = MagicMock()
    data_processor = DataProcessor(seq2seq_config.data, mock_tokenizer)
    train_dataset, eval_dataset = data_processor.create_dataset()

    mock_load_dataset.assert_called_once_with(seq2seq_config.data.format, data_files=seq2seq_config.data.path)
    assert train_dataset is not None
    assert eval_dataset is not None

@patch('llm_finetuning.training.base_trainer.HuggingFaceSeq2SeqTrainer')
@patch('llm_finetuning.training.base_trainer.Seq2SeqTrainingArguments')
def test_seq2seq_trainer_initialization(mock_training_arguments, mock_huggingface_trainer, seq2seq_config):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_train_dataset = MagicMock()
    mock_eval_dataset = MagicMock()

    trainer_instance = Trainer(mock_model, mock_tokenizer, mock_train_dataset, mock_eval_dataset, seq2seq_config.training)
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
