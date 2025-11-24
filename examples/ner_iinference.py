import torch
import yaml
import os
from transformers import AutoTokenizer, AutoConfig
from multitask_bert.core.config import ExperimentConfig
from multitask_bert.core.model import MultiTaskModel
from multitask_bert.tasks.ner import NERTask
import json
from typing import List, Dict, Any

class NERInference:
    def __init__(self, model_dir: str):
        """
        Initialize the NER inference with trained model. 
        
        Args:
            model_dir: Path to the directory containing best_model and experiment_config.yaml
        """
        self.model_dir = model_dir
        self.best_model_path = os.path.join(model_dir, "best_model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.best_model_path)
        self.model = self._load_model()
        
        # Get label map - FIXED: Handle the label mapping correctly
        self.id_to_label = self._get_label_map()
        
        print(f"Loaded model with {len(self.id_to_label)} labels: {self.id_to_label}")
    
    def _load_config(self) -> ExperimentConfig:
        """Load experiment configuration from saved YAML."""
        config_path = os.path.join(self.model_dir, "experiment_config.yaml")
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig(**config_dict)
    
    def _get_label_map(self):
        """Get the correct label mapping from config."""
        task_config = self.config.tasks[0]
        
        # Check if label_maps exists and has the correct structure
        if hasattr(task_config, 'label_maps') and task_config.label_maps:
            label_map = task_config.label_maps.get('ner_head', {})
            # If label_map has string keys, it's already id_to_label
            if label_map and isinstance(next(iter(label_map.keys())), str):
                return label_map
            # If label_map has integer keys, we need to invert it
            elif label_map and isinstance(next(iter(label_map.keys())), int):
                return {v: k for k, v in label_map.items()}
        
        # Fallback: Create default label map based on number of labels
        num_labels = task_config.heads[0].num_labels
        print(f"Warning: Using fallback label mapping with {num_labels} labels")
        return {i: f"LABEL_{i}" for i in range(num_labels)}
    
    def _load_model(self):
        """Load the trained model."""
        # Load model configuration
        model_config = AutoConfig.from_pretrained(self.best_model_path)
        
        # Load model
        print(f"Loading model from: {self.best_model_path}")
        model = MultiTaskModel(
            config=model_config,
            model_config=self.config.model,
            task_configs=self.config.tasks
        )
        
        # Load the state_dict manually, handling safetensors
        model_weights_path = os.path.join(self.best_model_path, "model.safetensors")
        if os.path.exists(model_weights_path):
            from safetensors.torch import load_file
            state_dict = load_file(model_weights_path)
        else:
            model_weights_path = os.path.join(self.best_model_path, "pytorch_model.bin")
            state_dict = torch.load(model_weights_path, map_location=self.device)
        
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text for inference."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.tokenizer.max_length,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def predict_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict named entities in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of entities with their text, label, and positions
        """
        # Preprocess text
        inputs = self.preprocess_text(text)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_id=0  # Assuming NER is the first task
            )
        
        # Get predictions
        logits = outputs["logits"]['ner_head'] # Changed to outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
        # Get tokens and their positions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Extract entities
        entities = self._extract_entities(text, tokens, predictions)
        return entities
    
    def _extract_entities(self, original_text: str, tokens: List[str], predictions: List[int]) -> List[Dict[str, Any]]:
        """Extract entities from predictions and map back to original text."""
        entities = []
        current_entity = None
        
        # Get character offsets for tokens
        encoding = self.tokenizer(original_text, return_offsets_mapping=True)
        offset_mapping = encoding['offset_mapping'][0]  # Get first (and only) sequence
        
        for i, (token, pred_idx, offset) in enumerate(zip(tokens, predictions, offset_mapping)):
            # Skip special tokens and padding
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            label = self.id_to_label.get(pred_idx, 'O')
            
            # Skip 'O' labels
            if label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            start_char, end_char = offset
            
            # If we're starting a new entity
            if current_entity is None or current_entity['label'] != label:
                if current_entity:
                    entities.append(current_entity)
                
                current_entity = {
                    'text': original_text[start_char:end_char],
                    'label': label,
                    'start': start_char,
                    'end': end_char
                }
            else:
                # Continue the current entity
                # Handle wordpiece tokens by checking if they're adjacent
                if start_char == current_entity['end'] or start_char == current_entity['end'] + 1:
                    current_entity['text'] += original_text[start_char:end_char]
                    current_entity['end'] = end_char
                else:
                    # Non-adjacent token, start new entity
                    entities.append(current_entity)
                    current_entity = {
                        'text': original_text[start_char:end_char],
                        'label': label,
                        'start': start_char,
                        'end': end_char
                    }
        
        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def debug_predictions(self, text: str):
        """Debug method to see raw predictions."""
        inputs = self.preprocess_text(text)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_id=0
            )
        
        logits = outputs["logits"]['ner_head'] # Changed to outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        print(f"\n🔍 Debug predictions for: {text}")
        print("Token -> Prediction -> Label")
        for token, pred in zip(tokens, predictions):
            if token not in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                label = self.id_to_label.get(pred, 'O')
                print(f"  {token} -> {pred} -> {label}")

def main():
    """Main function to test the inference."""
    # Path to your trained model directory
    model_dir = "./unified_results_ner"
    
    # Initialize inference
    print("Loading trained model...")
    ner_inference = NERInference(model_dir)
    
    # Test texts from your dataset
    test_texts = [
        "Hello, I'm Vincent from Dar es Salaam. I need help with my 10-year-old daughter.",
        "Mwangi Kennedy was seriously bruised and was taken to the Hospital in Nairobi.",
        "There's a 7-year-old boy named Asani who lives with us in Misungwi, Mwanza."
    ]
    
    print("\n" + "="*60)
    print("NER Inference Results - Swahili Child Helpline")
    print("="*60)
    
    # First, run debug to see what's happening
    print("\n🧪 Running debug predictions...")
    ner_inference.debug_predictions(test_texts[0])
    
    print("\n" + "="*60)
    print("Entity Extraction Results")
    print("="*60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Text {i}: {text}")
        entities = ner_inference.predict_entities(text)
        
        if entities:
            print("🔍 Entities found:")
            for entity in entities:
                print(f"   • '{entity['text']}' → {entity['label']} (chars {entity['start']}-{entity['end']})")
        else:
            print("   ❌ No entities detected.")

if __name__ == "__main__":
    main()
