
import yaml
import argparse
from seq2seq_models.core.config import Seq2SeqFinetuningConfig, ModelConfig, DataConfig, TrainingConfig, LoggingConfig, TeacherStudentConfig
from seq2seq_models.model.seq2seq_model import Seq2SeqModel
from seq2seq_models.data.data_processing import DataProcessor
from seq2seq_models.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the experiment config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Basic validation
    if not config_dict.get('model') or not config_dict['model'].get('name'):
        raise ValueError("Model configuration and model name are required.")
    if not config_dict.get('data') or not config_dict['data'].get('path'):
        raise ValueError("Data configuration and data path are required.")
    if not config_dict.get('training') or not config_dict['training'].get('output_dir'):
        raise ValueError("Training configuration and output directory are required.")

    teacher_student_config = TeacherStudentConfig(**config_dict['model']['teacher_student_config']) if config_dict['model'].get('teacher_student_config') else None
    model_config = ModelConfig(
        name=config_dict['model']['name'],
        teacher_student_config=teacher_student_config
    )
    data_config = DataConfig(**config_dict['data'])
    logging_config = LoggingConfig(**config_dict['training']['logging_config']) if config_dict['training'].get('logging_config') else None
    training_config = TrainingConfig(
        output_dir=config_dict['training']['output_dir'],
        learning_rate=config_dict['training']['learning_rate'],
        batch_size=config_dict['training']['batch_size'],
        num_epochs=config_dict['training']['num_epochs'],
        gradient_accumulation_steps=config_dict['training']['gradient_accumulation_steps'],
        logging_steps=config_dict['training']['logging_steps'],
        save_steps=config_dict['training']['save_steps'],
        logging_config=logging_config
    )
    
    config = Seq2SeqFinetuningConfig(
        model=model_config,
        data=data_config,
        training=training_config
    )

    seq2seq_model_instance = Seq2SeqModel(config.model)
    model, tokenizer = seq2seq_model_instance.create_model_and_tokenizer()
    
    data_processor = DataProcessor(config.data, tokenizer)
    train_dataset, eval_dataset = data_processor.create_dataset()
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=config.training,
    )

    trainer.train()

if __name__ == "__main__":
    main()
