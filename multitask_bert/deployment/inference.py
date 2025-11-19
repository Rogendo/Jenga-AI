import torch
from transformers import AutoTokenizer, AutoConfig
from typing import Dict, Any, List
from ..core.config import ExperimentConfig, TaskConfig
from ..core.model import MultiTaskModel
from ..tasks.base import BaseTask
from ..tasks.classification import SingleLabelClassificationTask, MultiLabelClassificationTask
from ..tasks.ner import NERTask

class InferenceHandler:
    """
    Handles inference for a trained MultiTaskModel.
    """
    def __init__(self, model_path: str, config: ExperimentConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Re-instantiate tasks from config
        tasks = [self._get_task_class(t.type)(t) for t in config.tasks]
        
        # Load model config
        model_config = AutoConfig.from_pretrained(model_path)
        
        self.model = MultiTaskModel(
            config=model_config,
            model_config=config.model,
            tasks=tasks
        )
        self.model.eval()
        self.model.to(config.training.device)
        
        self.task_map = {task.name: i for i, task in enumerate(tasks)}

    def _get_task_class(self, task_type: str) -> BaseTask:
        """Maps a task type string to its corresponding class."""
        if task_type == "single_label_classification":
            return SingleLabelClassificationTask
        elif task_type == "multi_label_classification":
            return MultiLabelClassificationTask
        elif task_type == "ner":
            return NERTask
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Makes predictions for all tasks on a given text.
        """
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer.max_length,
            return_tensors="pt"
        ).to(self.config.training.device)

        predictions = {}
        with torch.no_grad():
            for task_config in self.config.tasks:
                task_id = self.task_map[task_config.name]
                
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    task_id=task_id
                )
                
                # Interpret logits for each head
                task_predictions = {}
                for head_config in task_config.heads:
                    head_name = head_config.name
                    logits = outputs.logits[head_name].cpu().numpy()
                    
                    if task_config.type == "single_label_classification":
                        predicted_id = logits.argmax(axis=-1).item()
                        task_predictions[head_name] = predicted_id # Can map to label string if label_maps are available
                    elif task_config.type == "multi_label_classification":
                        predicted_labels = (logits > 0.5).astype(int).tolist() # Binary predictions
                        task_predictions[head_name] = predicted_labels
                    elif task_config.type == "ner":
                        # For NER, we need to decode token-level predictions
                        predicted_ids = logits.argmax(axis=-1).flatten().tolist()
                        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].flatten().tolist())
                        
                        decoded_entities = []
                        current_entity = []
                        current_label = None
                        
                        for token, pred_id in zip(tokens, predicted_ids):
                            label = task_config.label_maps['ner_head'][pred_id] if 'ner_head' in task_config.label_maps else str(pred_id)
                            
                            if label != 'O' and token not in self.tokenizer.all_special_tokens:
                                if current_label is None: # Start of a new entity
                                    current_entity.append(token)
                                    current_label = label
                                elif label == current_label: # Continuation of the same entity
                                    current_entity.append(token)
                                else: # New entity with different label
                                    decoded_entities.append({"entity": self.tokenizer.convert_tokens_to_string(current_entity), "label": current_label})
                                    current_entity = [token]
                                    current_label = label
                            elif current_entity: # End of an entity
                                decoded_entities.append({"entity": self.tokenizer.convert_tokens_to_string(current_entity), "label": current_label})
                                current_entity = []
                                current_label = None
                        
                        if current_entity: # Add last entity if any
                            decoded_entities.append({"entity": self.tokenizer.convert_tokens_to_string(current_entity), "label": current_label})
                        
                        task_predictions[head_name] = decoded_entities
                
                predictions[task_config.name] = task_predictions
        
        return predictions
