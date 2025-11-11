import torch
from transformers import BertTokenizer
from multitask_bert.tasks.classification import ClassificationTask
from multitask_bert.tasks.ner import NERTask
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.training.trainer import Trainer, TrainingArguments, MultiTaskDataset
from multitask_bert.core.model import MultiTaskBert

def main():
    # 1. Define Tasks
    sentiment_task = ClassificationTask(
        name="SwahiliSentiment",
        label_map={0: "Negative", 1: "Positive"}
    )
    ner_task = NERTask(
        name="SwahiliNER",
        label_map={0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC", 5: "B-ORG", 6: "I-ORG"}
    )
    tasks = [sentiment_task, ner_task]

    # 2. Load Tokenizer and DataProcessor
    model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    data_processor = DataProcessor(tokenizer=tokenizer, max_seq_length=128)

    # 3. Process Data
    # For this example, we'll use the same data for training and evaluation
    data_files = {
        "SwahiliSentiment": "examples/sentiment_data.csv",
        "SwahiliNER": "examples/ner_data.csv" # This will be skipped by the current processor
    }
    
    # The processor will currently only process the classification data
    # and skip the NER data due to the placeholder implementation.
    train_tasks_data = data_processor.create_tasks_data(tasks, data_files)
    eval_tasks_data = train_tasks_data 

    # 4. Create Datasets
    train_dataset = MultiTaskDataset(train_tasks_data, tasks)
    eval_dataset = MultiTaskDataset(eval_tasks_data, tasks)

    # 5. Instantiate Model
    # Note: The from_pretrained method will initialize the BERT part with pretrained weights
    # and the new heads with random weights.
    model = MultiTaskBert.from_pretrained(model_name, tasks=tasks)

    # 6. Define Training Arguments
    # We need to pass the pad_token_id to the arguments for the collate_fn
    args = TrainingArguments(
        epochs=3,
        batch_size=4,
        learning_rate=2e-5,
        pad_token_id=tokenizer.pad_token_id
    )

    # 7. Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # 8. Start Training
    print("Starting multi-task training...")
    trainer.train()
    print("Training complete.")

if __name__ == "__main__":
    main()
