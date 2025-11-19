
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from seq2seq_models.core.config import ModelConfig
from seq2seq_models.model.teacher_student import TeacherStudentModel

class Seq2SeqModel:
    """
    A factory class for creating Sequence-to-Sequence (Seq2Seq) models for fine-tuning.

    This class handles loading base Seq2Seq models from the Hugging Face Hub
    and setting up teacher-student distillation if configured.
    """
    def __init__(self, model_config: ModelConfig):
        """
        Initializes the Seq2SeqModel factory with a given model configuration.

        Args:
            model_config (ModelConfig): The configuration object defining the
                                        base model and its modifications.
        """
        self.model_config = model_config

    def create_model_and_tokenizer(self):
        """
        Creates and configures the Seq2Seq model and its tokenizer.

        Returns:
            tuple: A tuple containing the configured model and its tokenizer.
        Raises:
            ValueError: If the model or tokenizer cannot be loaded.
        """
        model_name = self.model_config.name
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except OSError as e:
            raise ValueError(f"Failed to load model '{model_name}'. Check model name, network connection, or Hugging Face credentials. Error: {e}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError as e:
            raise ValueError(f"Failed to load tokenizer for model '{model_name}'. Check model name, network connection, or Hugging Face credentials. Error: {e}")


        if self.model_config.teacher_student_config:
            teacher_model_name = self.model_config.teacher_student_config.teacher_model
            try:
                teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name)
            except OSError as e:
                raise ValueError(f"Failed to load teacher model '{teacher_model_name}'. Check model name, network connection, or Hugging Face credentials. Error: {e}")
            model = TeacherStudentModel(student_model=model, teacher_model=teacher_model)

        return model, tokenizer
