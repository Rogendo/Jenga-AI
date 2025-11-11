# Metrics

The Jenga-NLP framework provides a set of metric functions for evaluating the performance of your multi-task models. These functions are located in `multitask_bert/utils/metrics.py` and are used by the `Trainer` class during the evaluation process.

## Supported Metrics

The framework currently supports the following metrics:

### Classification Metrics

-   **Function:** `compute_classification_metrics(preds, labels)`
-   **Description:** This function computes the accuracy, precision, recall, and F1 score for single-label classification tasks.
-   **Arguments:**
    -   `preds`: The predictions from the model.
    -   `labels`: The true labels.
-   **Returns:** A dictionary of metrics.

### Multi-Label Metrics

-   **Function:** `compute_multi_label_metrics(preds, labels)`
-   **Description:** This function computes the accuracy, precision, recall, and F1 score for multi-label classification tasks.
-   **Arguments:**
    -   `preds`: The predictions from the model.
    -   `labels`: The true labels.
-   **Returns:** A dictionary of metrics.

### NER Metrics

-   **Function:** `compute_ner_metrics(preds, labels, label_map)`
-   **Description:** This function computes the accuracy, precision, recall, and F1 score for named entity recognition (NER) tasks. It uses the `seqeval` library to compute the metrics at the entity level.
-   **Arguments:**
    -   `preds`: The predictions from the model.
    -   `labels`: The true labels.
    -   `label_map`: A dictionary that maps label IDs to label names.
-   **Returns:** A dictionary of metrics.

## How Metrics are Calculated

During the evaluation process, the `Trainer` class iterates through each task's evaluation dataloader and collects the predictions and labels for each head. It then calls the appropriate metric function based on the task type to compute the metrics for that head.

The metrics for all heads are then aggregated into a single dictionary and logged to MLflow or TensorBoard.

## Custom Metrics

To use a custom metric function, you will need to modify the `evaluate` method of the `Trainer` class in `multitask_bert/training/trainer.py`.

1.  **Import your metric function:** Import your custom metric function at the top of the file.

2.  **Call your metric function:** In the `evaluate` method, after the predictions and labels have been collected, call your custom metric function with the predictions and labels.

3.  **Add the results to the `task_metrics` dictionary:** Add the results of your custom metric function to the `task_metrics` dictionary.

By following these steps, you can easily extend the framework to support any metric function you need.
