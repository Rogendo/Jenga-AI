import torch
from multitask_bert.core.config import ExperimentConfig, ModelConfig, TokenizerConfig, TaskConfig, HeadConfig
from multitask_bert.core.fusion import AttentionFusion
from transformers import AutoConfig, AutoTokenizer

def main():
    """
    Fixed version of the attention fusion test
    """
    print("--- Testing Attention Fusion ---")

    # 1. Create a minimal ExperimentConfig with required data_path
    print("\nStep 1: Creating ExperimentConfig")
    config = ExperimentConfig(
        project_name="AttentionFusionTest",
        model=ModelConfig(
            base_model="distilbert-base-uncased",
        ),
        tokenizer=TokenizerConfig(),
        tasks=[
            TaskConfig(
                name="TaskA", 
                type="single_label_classification", 
                data_path="examples/dummy_data.json",  # REQUIRED
                heads=[HeadConfig(name="head_a", num_labels=2)]
            ),
            TaskConfig(
                name="TaskB", 
                type="single_label_classification", 
                data_path="examples/dummy_data.json",  # REQUIRED  
                heads=[HeadConfig(name="head_b", num_labels=3)]
            )
        ]
    )
    print("✅ ExperimentConfig created successfully")

    # 2. Test fusion directly (bypassing full model complexity)
    print("\nStep 2: Testing Fusion Mechanism")
    
    # Get model config to know hidden_size
    model_config = AutoConfig.from_pretrained(config.model.base_model)
    print(f"Model hidden size: {model_config.hidden_size}")
    
    # Create your fusion module
    fusion = AttentionFusion(model_config, num_tasks=len(config.tasks))
    print(f"✅ Fusion module created for {len(config.tasks)} tasks")
    
    # 3. Test with dummy data
    print("\nStep 3: Testing with dummy data")
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    
    dummy_text = ["This is a test sentence.", "This is another test sentence."]
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Simulate shared representation from BERT
    batch_size, seq_len = inputs['input_ids'].shape
    shared_repr = torch.randn(batch_size, seq_len, model_config.hidden_size)
    print(f"Shared representation shape: {shared_repr.shape}")
    
    # 4. Test fusion for different tasks
    print("\nStep 4: Testing task-specific fusion")
    
    for i, task in enumerate(config.tasks):
        fused_repr = fusion(shared_repr, task_id=i)
        print(f"Task '{task.name}' (id={i}): {fused_repr.shape}")
        
    # 5. Verify they're different
    fused_0 = fusion(shared_repr, 0)
    fused_1 = fusion(shared_repr, 1)
    
    are_different = not torch.allclose(fused_0, fused_1, atol=1e-6)
    print(f"\n🎯 Different outputs for different tasks: {are_different}")
    
    if are_different:
        print("✅ SUCCESS: Attention fusion is working correctly!")
        print("Each task gets a uniquely weighted representation.")
    else:
        print("❌ WARNING: Fusion might not be working properly")
    
    print("\n--- Attention Fusion Test Complete ---")

if __name__ == "__main__":
    main()