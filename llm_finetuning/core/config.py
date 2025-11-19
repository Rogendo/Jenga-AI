
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class PeftConfig:
    """
    Configuration for Parameter-Efficient Fine-Tuning (PEFT) methods.
    """
    peft_type: str = "LORA"
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)

@dataclass
class TeacherStudentConfig:
    """
    Configuration for teacher-student knowledge distillation.
    """
    teacher_model: str
    distillation_alpha: float = 0.5

@dataclass
class ModelConfig:
    """
    Configuration for the base LLM and its modifications.
    """
    name: str
    quantization: Optional[str] = None
    peft_config: Optional[PeftConfig] = None
    teacher_student_config: Optional[TeacherStudentConfig] = None

@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    """
    path: str
    format: str = "jsonl"
    train_split: str = "train"
    eval_split: Optional[str] = None
    max_length: int = 512

@dataclass
class LoggingConfig:
    """
    Configuration for experiment logging and monitoring.
    """
    report_to: Optional[str] = None
    run_name: Optional[str] = None

@dataclass
class TrainingConfig:
    """
    Configuration for the training process.
    """
    output_dir: str
    learning_rate: float = 2e-4
    batch_size: int = 4
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    save_steps: int = 100
    logging_config: Optional[LoggingConfig] = None

@dataclass
class LLMFinetuningConfig:
    """
    Overall configuration for an LLM fine-tuning experiment.
    """
    model: ModelConfig
    data: List[DataConfig]
    training: TrainingConfig
