
import torch
from transformers import TrainingArguments, Trainer as HuggingFaceTrainer, Seq2SeqTrainingArguments, Seq2SeqTrainer as HuggingFaceSeq2SeqTrainer, DataCollatorForLanguageModeling
from typing import Optional

class BaseTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, training_config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_config = training_config
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # For causal LM

    def _get_training_arguments(self, is_seq2seq: bool = False):
        report_to = []
        run_name = None
        if self.training_config.logging_config:
            if self.training_config.logging_config.report_to:
                report_to.append(self.training_config.logging_config.report_to)
            run_name = self.training_config.logging_config.run_name

        common_args = {
            "output_dir": self.training_config.output_dir,
            "learning_rate": self.training_config.learning_rate,
            "per_device_train_batch_size": self.training_config.batch_size,
            "per_device_eval_batch_size": self.training_config.batch_size,
            "num_train_epochs": self.training_config.num_epochs,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "logging_steps": self.training_config.logging_steps,
            "save_steps": self.training_config.save_steps,
            "eval_strategy": "steps" if self.eval_dataset else "no",
            "eval_steps": self.training_config.save_steps if self.eval_dataset else None,
            "fp16": torch.cuda.is_available(),
            "report_to": report_to if report_to else None,
            "run_name": run_name,
        }

        if is_seq2seq:
            return Seq2SeqTrainingArguments(**common_args)
        else:
            return TrainingArguments(**common_args)

    def train(self, is_seq2seq: bool = False):
        training_args = self._get_training_arguments(is_seq2seq)

        if is_seq2seq:
            trainer_class = HuggingFaceSeq2SeqTrainer
        else:
            trainer_class = HuggingFaceTrainer
        
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.train()
