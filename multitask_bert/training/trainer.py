import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any
import itertools
import os

from ..core.config import ExperimentConfig
from ..core.model import MultiTaskModel
from ..utils.metrics import (
    compute_classification_metrics, 
    compute_multi_label_metrics, 
    compute_ner_metrics
)

# --- Task-Specific Collate Functions ---

def ner_collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100) # -100 is ignore_index for CrossEntropyLoss

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': padded_labels,
    }

def classification_collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    labels = {}
    first_item_labels = batch[0]['labels']
    if isinstance(first_item_labels, dict): # Multi-head case
        for key in first_item_labels.keys():
            labels[key] = torch.stack([item['labels'][key] for item in batch])
    else: # Single tensor case (single-label classification)
        labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': labels,
    }

# --- Trainer Class ---

class Trainer:
    """
    The main trainer class for the unified multi-task framework.
    Uses task-specific dataloaders and a round-robin iterator for training.
    """
    def __init__(self, config: ExperimentConfig, model: MultiTaskModel, tokenizer: PreTrainedTokenizer, 
                 train_datasets: Dict[str, Any], eval_datasets: Dict[str, Any]):
        self.config = config
        self.training_args = config.training
        self.model = model.to(self.training_args.device)
        self.tokenizer = tokenizer
        
        self.task_map = {task.name: i for i, task in enumerate(config.tasks)}
        
        self.train_dataloaders = self._create_dataloaders(train_datasets, is_eval=False)
        self.eval_dataloaders = self._create_dataloaders(eval_datasets, is_eval=True)
        
        self.logger = None
        self._init_logger()

    def _init_logger(self):
        if self.training_args.logging:
            service = self.training_args.logging.service
            exp_name = self.training_args.logging.experiment_name
            
            if service == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(self.training_args.output_dir, "logs", exp_name)
                self.logger = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard logger initialized. Log directory: {log_dir}")
            elif service == "mlflow":
                import mlflow
                if self.training_args.logging.tracking_uri:
                    mlflow.set_tracking_uri(self.training_args.logging.tracking_uri)
                mlflow.set_experiment(exp_name)
                mlflow.start_run()
                self.logger = mlflow
                print(f"MLflow logger initialized. Experiment: '{exp_name}'")

    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str):
        if self.logger:
            service = self.training_args.logging.service
            if service == "tensorboard":
                for key, value in metrics.items():
                    self.logger.add_scalar(f"{prefix}/{key}", value, step)
            elif service == "mlflow":
                self.logger.log_metrics(metrics, step=step)

    def _create_dataloaders(self, datasets: Dict[str, Any], is_eval: bool) -> Dict[str, DataLoader]:
        dataloaders = {}
        for task_name, dataset in datasets.items():
            task_config = next(t for t in self.config.tasks if t.name == task_name)
            
            collate_fn_to_use = None
            if task_config.type == 'ner':
                collate_fn_to_use = lambda batch: ner_collate_fn(batch, self.tokenizer)
            elif task_config.type == 'single_label_classification' or task_config.type == 'multi_label_classification':
                collate_fn_to_use = lambda batch: classification_collate_fn(batch, self.tokenizer)
            
            dataloaders[task_name] = DataLoader(
                dataset,
                batch_size=self.training_args.batch_size,
                shuffle=not is_eval,
                collate_fn=collate_fn_to_use
            )
        return dataloaders

    def _create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer = AdamW(self.model.parameters(), lr=self.training_args.learning_rate, weight_decay=self.training_args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.training_args.warmup_steps, num_training_steps=num_training_steps)
        return optimizer, scheduler

    def train(self):
        num_training_steps = sum(len(dl) for dl in self.train_dataloaders.values()) * self.training_args.num_epochs
        optimizer, scheduler = self._create_optimizer_and_scheduler(num_training_steps)
        
        progress_bar = tqdm(total=num_training_steps, desc="Training")
        global_step = 0

        best_metric = float('inf') if not self.training_args.greater_is_better else float('-inf')
        epochs_no_improve = 0

        for epoch in range(self.training_args.num_epochs):
            self.model.train()
            
            train_iterators = {name: iter(dl) for name, dl in self.train_dataloaders.items()}
            
            while train_iterators:
                for task_name, iterator in list(train_iterators.items()):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        del train_iterators[task_name]
                        continue

                    optimizer.zero_grad() 
                    
                    task_id = self.task_map[task_name]
                    
                    input_ids = batch['input_ids'].to(self.training_args.device)
                    attention_mask = batch['attention_mask'].to(self.training_args.device)
                    labels = batch['labels']
                    if isinstance(labels, dict):
                        labels = {k: v.to(self.training_args.device) for k, v in labels.items()}
                    else:
                        labels = labels.to(self.training_args.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_id=task_id,
                        labels=labels
                    )

                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    self._log_metrics({"train_loss": loss.item()}, global_step, "Train")
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item())
                    global_step += 1

            if self.eval_dataloaders:
                eval_metrics = self.evaluate()
                self._log_metrics(eval_metrics, epoch, "Eval")
                print(f"\nEpoch {epoch + 1} Eval Metrics: {eval_metrics}")
                
                if self.training_args.early_stopping_patience:
                    metric_to_check = eval_metrics.get(self.training_args.metric_for_best_model, None)
                    if metric_to_check is None:
                        print(f"Warning: Metric '{self.training_args.metric_for_best_model}' not found for early stopping.")
                    else:
                        if (self.training_args.greater_is_better and metric_to_check > best_metric) or \
                           (not self.training_args.greater_is_better and metric_to_check < best_metric):
                            best_metric = metric_to_check
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        
                        if epochs_no_improve >= self.training_args.early_stopping_patience:
                            print("Early stopping triggered.")
                            break
        progress_bar.close()

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        all_metrics = {}
        overall_f1_scores = []

        for task_name, dataloader in self.eval_dataloaders.items():
            task_config = next(t for t in self.config.tasks if t.name == task_name)
            
            all_preds = {head.name: [] for head in task_config.heads}
            all_labels = {head.name: [] for head in task_config.heads}
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                    task_id = self.task_map[task_name]
                    
                    input_ids = batch['input_ids'].to(self.training_args.device)
                    attention_mask = batch['attention_mask'].to(self.training_args.device)
                    labels = batch['labels']
                    if isinstance(labels, dict):
                        labels = {k: v.to(self.training_args.device) for k, v in labels.items()}
                    else:
                        labels = labels.to(self.training_args.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_id=task_id,
                        labels=labels
                    )
                    
                    for head_config in task_config.heads:
                        head_name = head_config.name
                        logits = outputs.logits[head_name].cpu().numpy()
                        all_preds[head_name].append(logits)
                        
                        if isinstance(labels, dict):
                            all_labels[head_name].append(labels[head_name].cpu().numpy())
                        else:
                            all_labels[head_name].append(labels.cpu().numpy())

            task_metrics = {}
            for head_config in task_config.heads:
                head_name = head_config.name
                preds_np = np.concatenate(all_preds[head_name], axis=0)
                labels_np = np.concatenate(all_labels[head_name], axis=0)

                if task_config.type == 'ner':
                    metrics = compute_ner_metrics(preds_np, labels_np, task_config.label_maps['ner_head'])
                elif task_config.type == 'multi_label_classification':
                    metrics = compute_multi_label_metrics(preds_np, labels_np)
                elif task_config.type == 'single_label_classification':
                    metrics = compute_classification_metrics(preds_np, labels_np)
                else:
                    metrics = {}
                
                for metric_name, value in metrics.items():
                    task_metrics[f"{task_name}_{head_name}_{metric_name}"] = value
                
                if 'f1' in metrics:
                    overall_f1_scores.append(metrics['f1'])

            all_metrics.update(task_metrics)
            
        if overall_f1_scores:
            all_metrics['eval_overall_f1'] = np.mean(overall_f1_scores)
            if not self.training_args.greater_is_better:
                all_metrics['eval_loss'] = -all_metrics['eval_overall_f1']
            else:
                all_metrics['eval_loss'] = 1 - all_metrics['eval_overall_f1']
        else:
            all_metrics['eval_loss'] = 0.0

        return all_metrics

    def close(self):
        if self.logger:
            service = self.training_args.logging.service
            if service == "tensorboard":
                self.logger.close()
            elif service == "mlflow":
                self.logger.end_run()
            print(f"{service} logger closed.")
