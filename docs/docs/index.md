#   Democratizing NLP

**Making Advanced NLP Accessible to Every African Developer**

[Get Started](getting-started/installation.md)

[View Gallery](gallery.md)

##  Why Democraticise NLP?

<div class="grid cards" markdown>
<ul>
<li>
    <b> 50% Faster Development </b> Reduce NLP setup from weeks to hours with our unified framework
</li>

<li> 
    <b>African Context Aware</b> Built-in support for Swahili, Luganda, and 10+ African languages
</li>

<li> 
    <b>Multi-Task Fusion </b> Combine models mathematically for efficient multi-task learning
</li>
</ul>
</div>

##  Quick Start

```python
from democraticise_nlp import NLPFramework

# Train Swahili sentiment analysis 

framework = NLPFramework("masakhane/swahili-bert")

framework.load_data("my_swahili_data.csv")
model = framework.train_task("sentiment")

result = model.predict("Habari za leo?")
print(result)

```