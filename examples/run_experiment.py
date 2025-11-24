import argparse
import dataclasses
import os
import yaml

# Import task modules to ensure tasks are registered
from multitask_bert.tasks import classification, ner
from multitask_bert.training.trainer import Trainer

def main(config_path: str):
    """
    Main function to run a multi-task experiment from a config file.
    """
    # 1. Instantiate Trainer directly from the config file
    print("🚀 Starting Jenga-AI Experiment...")
    trainer = Trainer.from_config(config_path)
    print("✅ Trainer initialized from config.")

    # 2. Start Training
    print("🔥 Starting training...")
    try:
        trainer.train()
        print("🎉 Training complete.")

        # 3. Final Evaluation
        print("Running final evaluation...")
        final_metrics = trainer.evaluate()
        print("Final evaluation metrics:")
        print(final_metrics)
    finally:
        # 4. Close the logger
        trainer.close()

    # 5. Save the updated config
    output_config_path = os.path.join(trainer.config.training.output_dir, "experiment_config.yaml")
    os.makedirs(trainer.config.training.output_dir, exist_ok=True)
    with open(output_config_path, 'w') as f:
        # Use the config from the trainer as it might have been updated
        yaml.dump(dataclasses.asdict(trainer.config), f, indent=2)
    print(f"✅ Updated experiment config saved to: {output_config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-task experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/experiment.yaml",
        help="Path to the experiment YAML file."
    )
    args = parser.parse_args()
    main(args.config)