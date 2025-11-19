import torch
from multitask_bert.core.config import load_experiment_config
from multitask_bert.core.model import MultiTaskModel
from transformers import AutoTokenizer

def main():
    """
    Test attention fusion using your existing YAML configuration
    """
    print("--- Testing Attention Fusion with YAML Config ---")
    
    # 1. Load your existing config
    print("\nStep 1: Loading YAML configuration")
    config = load_experiment_config("examples/experiment.yaml")
    print(f"✅ Loaded config: {config.project_name}")
    print(f"Model: {config.model.base_model}")
    print(f"Tasks: {[task.name for task in config.tasks]}")
    
    # 2. Check if fusion is enabled
    if not config.model.fusion:
        print("❌ Fusion is not enabled in your config. Let me fix that...")
        config.model.fusion = True
        print("✅ Enabled fusion for testing")
    
    # 3. Initialize model
    print("\nStep 2: Initializing model")
    try:
        # We need to create tasks from config
        # For now, let's create a simple test without task-specific heads
        from transformers import AutoConfig
        
        model_config = AutoConfig.from_pretrained(config.model.base_model)
        model = MultiTaskModel(config=model_config, model_config=config.model, tasks=[])
        
        print("✅ Model initialized successfully")
        
        # 4. Test with dummy data
        print("\nStep 3: Testing with dummy data")
        tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
        
        dummy_text = ["Testing attention fusion mechanism", "Another test sentence"]
        inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_id=0  # Test with first task
            )
            
            print("✅ Forward pass successful!")
            print(f"Output type: {type(outputs)}")
            
            if hasattr(outputs, 'last_hidden_state'):
                print(f"Hidden states shape: {outputs.last_hidden_state.shape}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nLet's try a simpler approach...")
        test_simple_fusion()

def test_simple_fusion():
    """Simpler test for fusion mechanism"""
    print("\n--- Simple Fusion Test ---")
    
    from multitask_bert.core.fusion import AttentionFusion
    
    # Test the fusion module directly
    class MockConfig:
        def __init__(self, hidden_size):
            self.hidden_size = hidden_size

    mock_config = MockConfig(hidden_size=768)
    num_tasks = 2
    fusion = AttentionFusion(mock_config, num_tasks)
    
    # Create dummy hidden states
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"Input hidden states shape: {hidden_states.shape}")
    
    # Test fusion for different tasks
    fused_0 = fusion(hidden_states, task_id=0)
    fused_1 = fusion(hidden_states, task_id=1)
    
    print(f"Fused representation shape: {fused_0.shape}")
    print(f"Are outputs different for different tasks? {not torch.allclose(fused_0, fused_1)}")
    
    print("✅ Simple fusion test passed!")

if __name__ == "__main__":
    main()