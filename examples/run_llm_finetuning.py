
import yaml
import argparse
from llm_finetuning.core.config import LLMFinetuningConfig, ModelConfig, DataConfig, TrainingConfig, PeftConfig, TeacherStudentConfig, LoggingConfig
from llm_finetuning.pipeline import FinetuningPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the experiment config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    peft_config = PeftConfig(**config_dict['model']['peft_config']) if config_dict['model'].get('peft_config') else None
    teacher_student_config = TeacherStudentConfig(**config_dict['model']['teacher_student_config']) if config_dict['model'].get('teacher_student_config') else None
    model_config = ModelConfig(
        name=config_dict['model']['name'],
        quantization=config_dict['model'].get('quantization'),
        peft_config=peft_config,
        teacher_student_config=teacher_student_config
    )
    
    if isinstance(config_dict['data'], list):
        data_configs = [DataConfig(**data_config) for data_config in config_dict['data']]
    else:
        data_configs = [DataConfig(**config_dict['data'])]

    logging_config = LoggingConfig(**config_dict['training']['logging_config']) if config_dict['training'].get('logging_config') else None
    training_config = TrainingConfig(
        output_dir=config_dict['training']['output_dir'],
        learning_rate=float(config_dict['training']['learning_rate']),
        batch_size=config_dict['training']['batch_size'],
        num_epochs=config_dict['training']['num_epochs'],
        gradient_accumulation_steps=config_dict['training']['gradient_accumulation_steps'],
        logging_steps=config_dict['training']['logging_steps'],
        save_steps=config_dict['training']['save_steps'],
        logging_config=logging_config
    )
    
    config = LLMFinetuningConfig(
        model=model_config,
        data=data_configs,
        training=training_config
    )

    pipeline = FinetuningPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
