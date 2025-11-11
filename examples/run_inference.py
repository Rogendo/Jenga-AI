import argparse
from multitask_bert.core.config import load_experiment_config
from multitask_bert.deployment.inference import InferenceHandler

def main(config_path: str):
    """
    Main function to run inference using a trained multi-task model.
    """
    # 1. Load Config
    print("Loading experiment configuration...")
    config = load_experiment_config(config_path)

    # 2. Instantiate InferenceHandler
    # For demonstration, we'll use the base model as a placeholder.
    # In a real scenario, this would be the path to your fine-tuned model.
    print(f"Loading model for inference from: {config.model.base_model}")
    inference_handler = InferenceHandler(model_path=config.model.base_model, config=config)

    # 3. Provide sample text inputs
    sample_texts = [
        "This is a great product, I love it!",
        "I am very disappointed with the service.",
        "John Doe works at Google in New York.",
        "The meeting will be held at KICC tomorrow.",
        "Hello, thank you for calling. How may I help you today?",
        "I have a problem with my internet connection. It's very slow."
    ]

    # 4. Make predictions
    print("\nMaking predictions on sample texts:")
    for text in sample_texts:
        print(f"\n--- Text: '{text}' ---")
        predictions = inference_handler.predict(text)
        for task_name, task_preds in predictions.items():
            print(f"  Task: {task_name}")
            for head_name, head_preds in task_preds.items():
                print(f"    Head '{head_name}': {head_preds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for a multi-task model.")
    parser.add_argument(
        "--config",
        type=str,
        default="unified_results/experiment_config.yaml", # Changed default config path
        help="Path to the experiment YAML file."
    )
    args = parser.parse_args()
    main(args.config)
