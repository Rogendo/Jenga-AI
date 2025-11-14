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
    use_offset_mapping_for_ner: bool = True

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
    model: ModelConfig = field(default_factory=ModelModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Loads a YAML config file and parses it into ExperimentConfig dataclasses."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Manually parse ModelConfig
    model_config_dict = config_dict.get('model', {})
    model_config = ModelConfig(
        base_model=model_config_dict.get('base_model', "distilbert-base-uncased"),
        dropout=model_config_dict.get('dropout', 0.1),
        fusion=model_config_dict.get('fusion')
    )

    # Manually parse TokenizerConfig
    tokenizer_config_dict = config_dict.get('tokenizer', {})
    tokenizer_config = TokenizerConfig(
        max_length=tokenizer_config_dict.get('max_length', 128),
        padding=tokenizer_config_dict.get('padding', "max_length"),
        truncation=tokenizer_config_dict.get('truncation', True),
        pad_token_id=tokenizer_config_dict.get('pad_token_id')
    )

    # Manually parse TrainingConfig
    training_config_dict = config_dict.get('training', {})
    logging_config_dict = training_config_dict.get('logging', {})
    logging_config = None
    if logging_config_dict:
        logging_config = LoggingConfig(
            service=logging_config_dict.get('service', "tensorboard"),
            experiment_name=logging_config_dict.get('experiment_name', "multitask_experiment"),
            tracking_uri=logging_config_dict.get('tracking_uri')
        )

    training_config = TrainingConfig(
        output_dir=training_config_dict.get('output_dir', "./results"),
        learning_rate=training_config_dict.get('learning_rate', 2.0e-5),
        batch_size=training_config_dict.get('batch_size', 16),
        num_epochs=training_config_dict.get('num_epochs', 3),
        weight_decay=training_config_dict.get('weight_decay', 0.01),
        warmup_steps=training_config_dict.get('warmup_steps', 100),
        eval_strategy=training_config_dict.get('eval_strategy', "epoch"),
        save_strategy=training_config_dict.get('save_strategy', "epoch"),
        load_best_model_at_end=training_config_dict.get('load_best_model_at_end', True),
        metric_for_best_model=training_config_dict.get('metric_for_best_model', "eval_loss"),
        greater_is_better=training_config_dict.get('greater_is_better', False),
        early_stopping_patience=training_config_dict.get('early_stopping_patience'),
        device=training_config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        logging=logging_config
    )

    # Manually parse TaskConfig and HeadConfig
    tasks_list = []
    for task_item in config_dict.get('tasks', []):
        if isinstance(task_item, dict):
            heads_list = []
            for head_item in task_item.get('heads', []):
                if isinstance(head_item, dict):
                    heads_list.append(HeadConfig(**head_item))
                else:
                    heads_list.append(head_item) # Append as is if not a dict

            tasks_list.append(TaskConfig(
                name=task_item.get('name'),
                type=task_item.get('type'),
                data_path=task_item.get('data_path'),
                heads=heads_list,
                label_maps=task_item.get('label_maps'),
                use_offset_mapping_for_ner=task_item.get('use_offset_mapping_for_ner', True)
            ))
        else:
            tasks_list.append(task_item) # Append as is if not a dict

    return ExperimentConfig(
        project_name=config_dict.get('project_name'),
        tasks=tasks_list,
        model=model_config,
        tokenizer=tokenizer_config,
        training=training_config
    )