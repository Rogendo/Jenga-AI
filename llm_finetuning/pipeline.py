
from llm_finetuning.core.config import LLMFinetuningConfig
from llm_finetuning.model.model_factory import ModelFactory
from llm_finetuning.data.data_processing import DataProcessor
from llm_finetuning.training.trainer import Trainer

class FinetuningPipeline:
    """
    Orchestrates the entire Large Language Model (LLM) fine-tuning process.

    This class acts as the central control flow for an LLM fine-tuning experiment,
    handling the initialization of the model, tokenizer, data processing, and trainer
    based on a provided configuration.
    """
    def __init__(self, config: LLMFinetuningConfig):
        """
        Initializes the FinetuningPipeline with a given configuration.

        Args:
            config (LLMFinetuningConfig): The complete configuration object
                                          defining the fine-tuning experiment.
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """
        Performs basic validation of the provided LLMFinetuningConfig.
        Raises ValueError if any critical configuration is missing or invalid.
        """
        if not self.config.model or not self.config.model.name:
            raise ValueError("Model configuration and model name are required.")
        if not self.config.data:
            raise ValueError("At least one data configuration is required.")
        for data_config in self.config.data:
            if not data_config.path:
                raise ValueError("Data path is required for all data configurations.")
        if not self.config.training or not self.config.training.output_dir:
            raise ValueError("Training configuration and output directory are required.")

    def run(self):
        """
        Executes the LLM fine-tuning workflow.

        This method performs the following steps:
        1. Creates the model and tokenizer using the ModelFactory.
        2. Prepares the datasets using the DataProcessor.
        3. Initializes the trainer.
        4. Starts the training process.
        """
        # 1. Create model
        model_factory = ModelFactory(self.config.model)
        model, tokenizer = model_factory.create_model()

        # 2. Create dataset
        data_processor = DataProcessor(self.config.data, tokenizer)
        train_dataset, eval_dataset = data_processor.create_dataset()

        # 3. Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=self.config.training,
        )

        # 4. Run training
        trainer.train()
