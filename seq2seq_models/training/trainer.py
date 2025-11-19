
from seq2seq_models.core.config import TrainingConfig
from llm_finetuning.training.base_trainer import BaseTrainer # Assuming base_trainer is in llm_finetuning for now

class Trainer(BaseTrainer):
    """
    A wrapper class around the Hugging Face `Seq2SeqTrainer` for Sequence-to-Sequence (Seq2Seq) fine-tuning.

    This class handles the configuration and execution of the training loop,
    including evaluation and integration with logging platforms like MLflow or TensorBoard.
    """
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, training_config: TrainingConfig):
        """
        Initializes the Trainer.

        Args:
            model: The Seq2Seq model to be trained.
            tokenizer: The tokenizer associated with the model.
            train_dataset: The dataset for training.
            eval_dataset: The dataset for evaluation (optional).
            training_config (TrainingConfig): The configuration object defining
                                              the training parameters.
        """
        super().__init__(model, tokenizer, train_dataset, eval_dataset, training_config)

    def train(self):
        """
        Configures and executes the training process using the Hugging Face `Seq2SeqTrainer`.

        This method sets up `Seq2SeqTrainingArguments` based on the provided configuration,
        including logging preferences, and then initiates the training.
        """
        super().train(is_seq2seq=True)
