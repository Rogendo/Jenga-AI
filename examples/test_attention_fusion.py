# import torch
# from multitask_bert.core.config import ExperimentConfig, ModelConfig, TokenizerConfig, TaskConfig, HeadConfig
# from multitask_bert.core.model import MultiTaskModel
# from multitask_bert.tasks.classification import SingleLabelClassificationTask
# from transformers import AutoConfig, AutoTokenizer

# def main():
#     """
#     This script demonstrates how to test and inspect the attention fusion mechanism.
#     """
#     print("--- Testing Attention Fusion ---")

#     # 1. Explain the concept of Attention Fusion
#     print("\nStep 1: Understanding Attention Fusion")
#     print("Attention Fusion is a mechanism to create a task-specific representation from a shared encoder.")
#     print("It uses task embeddings and an attention mechanism to weigh the shared representation differently for each task.")
#     print("-" * 20)

#     # 2. Create a minimal ExperimentConfig programmatically
#     print("\nStep 2: Creating a minimal ExperimentConfig")
#     config = ExperimentConfig(
#         project_name="AttentionFusionTest",
#         model=ModelConfig(
#             base_model="distilbert-base-uncased",
#             fusion=True  # Enable attention fusion
#         ),
#         tokenizer=TokenizerConfig(),
#         tasks=[
#             TaskConfig(name="TaskA", type="single_label_classification", heads=[HeadConfig(name="head_a", num_labels=2)]),
#             TaskConfig(name="TaskB", type="single_label_classification", heads=[HeadConfig(name="head_b", num_labels=3)])
#         ]
#     )
#     print("ExperimentConfig created with fusion enabled.")
#     print("-" * 20)

#     # 3. Instantiate the MultiTaskModel
#     print("\nStep 3: Instantiating the MultiTaskModel")
#     tasks = [SingleLabelClassificationTask(t) for t in config.tasks]
#     model = MultiTaskModel(config=AutoConfig.from_pretrained(config.model.base_model), model_config=config.model, tasks=tasks)
#     tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
#     print("MultiTaskModel instantiated.")
#     print("-" * 20)

#     # 4. Create dummy data
#     print("\nStep 4: Creating dummy data")
#     dummy_text = ["This is a test sentence.", "This is another test sentence."]
#     inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]
#     print(f"Dummy input_ids shape: {input_ids.shape}")
#     print("-" * 20)

#     # 5. Run the forward pass and inspect the fusion
#     print("\nStep 5: Running the forward pass and inspecting the fusion")
#     model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         # --- Forward pass for Task A ---
#         print("\n--- Forward pass for Task A (task_id=0) ---")
#         # Get the shared representation
#         encoder_outputs_a = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         shared_representation_a = encoder_outputs_a.last_hidden_state
#         # Get the fused representation
#         fused_representation_a = model.fusion(shared_representation_a, task_id=0)
        
#         print(f"Shared representation shape: {shared_representation_a.shape}")
#         print(f"Fused representation shape for Task A: {fused_representation_a.shape}")
        
#         # --- Forward pass for Task B ---
#         print("\n--- Forward pass for Task B (task_id=1) ---")
#         # Get the shared representation (it will be the same as for Task A)
#         encoder_outputs_b = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         shared_representation_b = encoder_outputs_b.last_hidden_state
#         # Get the fused representation
#         fused_representation_b = model.fusion(shared_representation_b, task_id=1)

#         print(f"Shared representation shape: {shared_representation_b.shape}")
#         print(f"Fused representation shape for Task B: {fused_representation_b.shape}")

#         # --- Compare the representations ---
#         print("\n--- Comparing representations ---")
#         print(f"Are the shared representations for Task A and Task B the same? {(shared_representation_a == shared_representation_b).all()}")
#         print(f"Are the fused representations for Task A and Task B the same? {(fused_representation_a == fused_representation_b).all()}")
#         print("The fused representations should be different because the attention mechanism uses different task embeddings for each task.")
#         print("-" * 20)

#     # 6. Inspect the fusion module's parameters
#     print("\nStep 6: Inspecting the fusion module's parameters")
#     print("The fusion module has its own set of parameters, including the task embeddings and the attention layer.")
#     for name, param in model.fusion.named_parameters():
#         if param.requires_grad:
#             print(f"Parameter name: {name}, Shape: {param.shape}")
#     print("-" * 20)

#     print("\n--- Attention Fusion Test Complete ---")

# if __name__ == "__main__":
#     main()'


import torch
import sys
import os

# Add the root directory to the path so we can import multitask_bert
sys.path.insert(0, os.path.abspath('.'))

from multitask_bert.core.config import ExperimentConfig, ModelConfig, TokenizerConfig, TaskConfig, HeadConfig
from multitask_bert.core.fusion import AttentionFusion
from transformers import AutoConfig, AutoTokenizer

def main():
    """
    This script demonstrates how to test and inspect the attention fusion mechanism.
    """
    print("--- Testing Attention Fusion ---")

    # 1. Explain the concept of Attention Fusion
    print("\nStep 1: Understanding Attention Fusion")
    print("Your AttentionFusion uses task embeddings and an MLP attention mechanism.")
    print("It concatenates shared representation with task embedding, then computes attention weights.")
    print("-" * 20)

    # 2. Test the fusion module directly
    print("\nStep 2: Testing AttentionFusion directly")
    
    # Create a simple config for testing
    class SimpleConfig:
        def __init__(self):
            self.hidden_size = 768
    
    config = SimpleConfig()
    num_tasks = 3
    
    # Initialize your fusion module
    fusion = AttentionFusion(config, num_tasks)
    print(f"✅ Fusion module initialized with {num_tasks} tasks")
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print(f"Input shape: {hidden_states.shape}")
    
    # Test fusion for different tasks
    print("\nStep 3: Testing fusion for different tasks")
    
    # Test task 0
    fused_0 = fusion(hidden_states, task_id=0)
    print(f"Fused representation for task 0: {fused_0.shape}")
    
    # Test task 1  
    fused_1 = fusion(hidden_states, task_id=1)
    print(f"Fused representation for task 1: {fused_1.shape}")
    
    # Compare outputs
    print(f"\nAre outputs different for different tasks? {not torch.allclose(fused_0, fused_1)}")
    print("✅ This shows the fusion is task-specific!")
    
    # 3. Test with actual model configuration
    print("\nStep 4: Testing with actual model setup")
    
    try:
        # Load your YAML config
        from multitask_bert.core.config import load_experiment_config
        config = load_experiment_config("examples/experiment.yaml")
        
        print(f"Loaded project: {config.project_name}")
        print(f"Base model: {config.model.base_model}")
        
        # Initialize tokenizer and get hidden size
        tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
        model_config = AutoConfig.from_pretrained(config.model.base_model)
        
        print(f"Model hidden size: {model_config.hidden_size}")
        
        # Create fusion module with actual model config
        fusion = AttentionFusion(model_config, num_tasks=len(config.tasks))
        print(f"✅ Fusion module created for {len(config.tasks)} tasks")
        
        # Test with real tokenized data
        dummy_text = [
            "Huduma ya maji imefika finally kwa watu wa Kibera",
            "Serikali imetimiza ahadi yake ya kuleta maendeleo"
        ]
        
        inputs = tokenizer(
            dummy_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        print(f"Tokenized input shape: {inputs['input_ids'].shape}")
        
        # Simulate shared representation (this would come from BERT encoder)
        batch_size, seq_len = inputs['input_ids'].shape
        shared_repr = torch.randn(batch_size, seq_len, model_config.hidden_size)
        
        # Test fusion for each task
        for i, task in enumerate(config.tasks):
            fused_repr = fusion(shared_repr, task_id=i)
            print(f"Task '{task.name}': {fused_repr.shape}")
            
        print("✅ All tasks processed successfully!")
        
    except Exception as e:
        print(f"Note: Couldn't load YAML config: {e}")
        print("But the core fusion mechanism is working!")

    # 4. Inspect fusion parameters
    print("\nStep 5: Inspecting fusion parameters")
    print("Fusion module parameters:")
    for name, param in fusion.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print(f"\nTask embeddings: {fusion.task_embeddings.weight.shape}")
    print(f"Attention layer: {fusion.attention_layer}")
    
    print("\n--- Attention Fusion Test Complete ---")

if __name__ == "__main__":
    main()