import torch
from dataclasses import dataclass, field
import dataclasses
from typing import List, Optional, Dict, Any
import yaml

@dataclass
class HeadConfig:
    """Configuration for a single prediction head."""
    name: str
    num_labels: int
    weight: float = 1.0

@dataclass
class TaskConfig:
    """Configuration for a single task."""
    name: str
    type: str  # 'classification', 'ner', 'multi_label_classification'
    data_path: str
    heads: List[HeadConfig]
    label_maps: Optional[Dict[str, Dict[int, str]]] = None

@dataclass
class FusionConfig:
    """Configuration for the fusion layer."""
    type: str = "attention"
    hidden_size: int = 768

@dataclass
class ModelConfig:
    """Configuration for the model."""
    base_model: str = "distilbert-base-uncased"
    dropout: float = 0.1
    fusion: Optional[FusionConfig] = None

    def __post_init__(self):
        if isinstance(self.fusion, dict):
            self.fusion = FusionConfig(**self.fusion)

@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""
    max_length: int = 128
    padding: str = "max_length"
    truncation: bool = True
    pad_token_id: Optional[int] = None # Added for dynamic update

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    service: str = "tensorboard" # 'mlflow' or 'tensorboard'
    experiment_name: str = "multitask_experiment"
    tracking_uri: Optional[str] = None # For MLflow

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    output_dir: str = "./results"
    learning_rate: float = 2.0e-5
    batch_size: int = 16
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: Optional[int] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging: Optional[LoggingConfig] = None

    def __post_init__(self):
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)

@dataclass
class ExperimentConfig:
    """Top-level configuration for an experiment."""
    project_name: str
    tasks: List[TaskConfig]
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # Recursively convert dicts to dataclasses
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.tokenizer, dict):
            self.tokenizer = TokenizerConfig(**self.tokenizer)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.tasks, list):
            self.tasks = [TaskConfig(**task) if isinstance(task, dict) else task for task in self.tasks]
            for task in self.tasks:
                if isinstance(task.heads, list):
                    task.heads = [HeadConfig(**head) if isinstance(head, dict) else head for head in task.heads]


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Loads a YAML config file and parses it into ExperimentConfig dataclasses."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ExperimentConfig(**config_dict)