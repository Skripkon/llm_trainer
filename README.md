# LLM Trainer in 5 lines of code

```python
from llm_trainer import create_dataset, Trainer

create_dataset(save_dir="data") # Create default dataset (FineWeb)
model = ...                     # Define your model
trainer = Trainer(model=model)  # use default parameters
trainer.train()                 # use created dataset
```
