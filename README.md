# `llm_trainer` in 5 Lines of Code

```python
from llm_trainer import create_dataset, LLMTrainer

create_dataset(save_dir="data")   # Generate the default FineWeb dataset
model = ...                       # Define or load your model
trainer = LLMTrainer(model)       # Initialize trainer with default settings
trainer.train()                   # Start training on the dataset
```
