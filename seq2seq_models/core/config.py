
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class TeacherStudentConfig:
    """
    Configuration for teacher-student knowledge distillation for Seq2Seq models.
    """
    teacher_model: str
    distillation_alpha: float = 0.5

@dataclass
class ModelConfig:
    """
    Configuration for the base Seq2Seq model and its modifications.
    """
    name: str
    teacher_student_config: Optional[TeacherStudentConfig] = None

@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing for Seq2Seq models.
    """
    path: str
    format: str = "json"
    source_lang: str = "en"
    target_lang: str = "fr"
    train_split: str = "train"
    eval_split: Optional[str] = None
    max_length: int = 128

@dataclass
class LoggingConfig:
    """
    Configuration for experiment logging and monitoring for Seq2Seq models.
    """
    report_to: Optional[str] = None
    run_name: Optional[str] = None

@dataclass
class TrainingConfig:
    """
    Configuration for the training process of Seq2Seq models.
    """
    output_dir: str
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    save_steps: int = 100
    logging_config: Optional[LoggingConfig] = None

@dataclass
class Seq2SeqFinetuningConfig:
    """
    Overall configuration for a Seq2Seq fine-tuning experiment.
    """
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
