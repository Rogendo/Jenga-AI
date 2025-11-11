import argparse
from transformers import AutoTokenizer, AutoConfig
import dataclasses # Added import
from multitask_bert.core.config import load_experiment_config
from multitask_bert.core.model import MultiTaskModel
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.training.trainer import Trainer
from multitask_bert.tasks.base import BaseTask
from multitask_bert.tasks.classification import SingleLabelClassificationTask, MultiLabelClassificationTask
from multitask_bert.tasks.ner import NERTask

def get_task_class(task_type: str) -> BaseTask:
    """Maps a task type string to its corresponding class."""
    if task_type == "single_label_classification":
        return SingleLabelClassificationTask
    elif task_type == "multi_label_classification":
        return MultiLabelClassificationTask
    elif task_type == "ner":
        return NERTask
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def main(config_path: str):
    """
    Main function to run a multi-task experiment from a config file.
    """
    # 1. Load Config
    print("Loading experiment configuration...")
    config = load_experiment_config(config_path)

    # 2. Load Tokenizer
    print(f"Loading tokenizer: {config.model.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Resize model embeddings if a new token was added
        # This will be handled when we init the model
    config.tokenizer.pad_token_id = tokenizer.pad_token_id


    # 3. Process Data
    print("Processing data for all tasks...")
    train_datasets, eval_datasets, updated_config = DataProcessor(config, tokenizer).process()
    config = updated_config # The processor might update num_labels, etc.

    # 4. Instantiate Tasks and Model
    print("Instantiating tasks and model...")
    tasks = [get_task_class(t.type)(t) for t in config.tasks]
    
    # Load the base model configuration
    model_config = AutoConfig.from_pretrained(config.model.base_model)

    model = MultiTaskModel.from_pretrained(
        config.model.base_model,
        config=model_config, # Pass the loaded config object
        model_config=config.model,
        tasks=tasks
    )
    # Resize embeddings if tokenizer was expanded
    model.resize_token_embeddings(len(tokenizer))

    # 5. Instantiate Trainer
    print("Instantiating trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets
    )

    # 6. Start Training
    print("Starting training...")
    try:
        trainer.train()
        print("Training complete.")

        # 7. Final Evaluation
        print("Running final evaluation...")
        final_metrics = trainer.evaluate()
        print("Final evaluation metrics:")
        print(final_metrics)
    finally:
        # 8. Close the logger
        trainer.close()

    # 9. Save the updated config
    import os
    import yaml
    output_config_path = os.path.join(config.training.output_dir, "experiment_config.yaml")
    os.makedirs(config.training.output_dir, exist_ok=True)
    with open(output_config_path, 'w') as f:
        yaml.dump(dataclasses.asdict(config), f, indent=2)
    print(f"Updated experiment config saved to: {output_config_path}")

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
