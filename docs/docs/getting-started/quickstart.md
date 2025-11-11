# Quick Start

## Your First Swahili Sentiment Model

```python
from democraticise_nlp import NLPFramework

# Initialize with African context
framework = NLPFramework(
    base_model="masakhane/swahili-bert",
    language_context="swahili"
)

# Load your data
framework.load_data("swahili_reviews.csv")

# Train sentiment analysis
model = framework.train_task("sentiment")

# Make predictions
predictions = model.predict("Bidii yako imenivutia sana!")
print(predictions)

```