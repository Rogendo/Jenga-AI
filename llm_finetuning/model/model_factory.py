
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig as PeftLoraConfig

from llm_finetuning.core.config import ModelConfig
from llm_finetuning.model.teacher_student import TeacherStudentModel

class ModelFactory:
    """
    A factory class for creating Large Language Models (LLMs) for fine-tuning.

    This class handles loading base models from the Hugging Face Hub,
    applying quantization, configuring Parameter-Efficient Fine-Tuning (PEFT)
    methods like LoRA, and setting up teacher-student distillation.
    """
    def __init__(self, model_config: ModelConfig):
        """
        Initializes the ModelFactory with a given model configuration.

        Args:
            model_config (ModelConfig): The configuration object defining the
                                        base model and its modifications.
        """
        self.model_config = model_config

    def create_model(self):
        """
        Creates and configures the LLM model and its tokenizer.

        Returns:
            tuple: A tuple containing the configured model and its tokenizer.
        Raises:
            ValueError: If the model or tokenizer cannot be loaded.
        """
        model_name = self.model_config.name
        quantization = self.model_config.quantization

        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except OSError as e:
            raise ValueError(f"Failed to load model '{model_name}'. Check model name, network connection, or Hugging Face credentials. Error: {e}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
        except OSError as e:
            raise ValueError(f"Failed to load tokenizer for model '{model_name}'. Check model name, network connection, or Hugging Face credentials. Error: {e}")


        if self.model_config.peft_config:
            peft_config = self.model_config.peft_config
            if peft_config.peft_type == "LORA":
                peft_lora_config = PeftLoraConfig(
                    r=peft_config.r,
                    lora_alpha=peft_config.lora_alpha,
                    lora_dropout=peft_config.lora_dropout,
                    target_modules=peft_config.target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, peft_lora_config)
                model.print_trainable_parameters()
            # Add other PEFT methods here
            else:
                raise ValueError(f"Unsupported PEFT type: {peft_config.peft_type}")

        if self.model_config.teacher_student_config:
            teacher_model_name = self.model_config.teacher_student_config.teacher_model
            try:
                teacher_model = AutoModelForCausalLM.from_pretrained(
                    teacher_model_name,
                    device_map="auto",
                )
            except OSError as e:
                raise ValueError(f"Failed to load teacher model '{teacher_model_name}'. Check model name, network connection, or Hugging Face credentials. Error: {e}")
            model = TeacherStudentModel(student_model=model, teacher_model=teacher_model)

        return model, tokenizer
