<h1 align="center">
<img src="https://github.com/Skripkon/llm_trainer/blob/main/assets/llm_trainer_logo.png?raw=true" style="width: 800">
</h1>

<div style="display: flex; align-items: center; gap: 30px;">
    <a href="https://pepy.tech/projects/llm-trainer" target="_blank">
        <img src="https://static.pepy.tech/badge/llm-trainer" height="30" alt="PyPI Downloads">
    </a>
    <a href="https://github.com/Skripkon/PokerBots/blob/main/LICENSE" target="_blank">
        <img src="https://raw.githubusercontent.com/Skripkon/PokerBots/27bba4cc02db1a785a9c6623f807f7e138ebbbf7/PokerBots/images/MIT_license.svg" height="30" alt="MIT License">
    </a>
</div>

# `llm_trainer` in 5 Lines of Code

```python
from llm_trainer import create_dataset, LLMTrainer

create_dataset(save_dir="data")   # Generate the default FineWeb dataset
model = ...                       # Define or load your model (GPT, xLSTM, Mamba...)
trainer = LLMTrainer(model)       # Initialize trainer with default settings
trainer.train(data_dir="data")    # Start training on the dataset
```

🔴 YouTube Video: [Train LLMs in code, spelled out](https://youtu.be/tFyDICExbHg)

> [!NOTE]
> Explore [usage examples](https://github.com/Skripkon/llm_trainer/blob/main/examples)

# Installation

```bash
$ pip install llm-trainer
```

# How to Prepare Data
 
## Option 1: Use the Default FineWeb Dataset  

```python
from llm_trainer import create_dataset

create_dataset(save_dir="data",         # Where to save created dataset
               chunks_limit=1_500,      # Maximum number of files (chunks) with tokens to create
               chunk_size=int(1e6))     # Number of tokens per chunk
```

## Option 2: Use your own data

1. Your dataset should be structured as a JSON array, where each entry contains a "text" field.
You can store your data in one or multiple JSON files.

Example JSON file:
```
[
   {"text": "Learn about LLMs: https://www.youtube.com/@_NickTech"},
   {"text": "Open-source python library to train LLMs: https://github.com/Skripkon/llm_trainer."},
   {"text": "My name is Nikolay Skripko. Hello from Russia (2025)."}
]
```

2. Run the following code to convert your JSON files into a tokenized dataset:

```python
from llm_trainer import create_dataset_from_json

create_dataset_from_json(save_dir="data",        # Where to save created dataset
                         json_dir="json_files",  # Path to your JSON files
                         chunks_limit = 1_500,   # Maximum number of files (chunks) with tokens to create
                         chunk_size=int(1e6))    # Number of tokens per chunk 
```

# Which Models Are Valid?

You can train **ANY** LLM that expects a tensor `X` with shape `(batch_size, context_window)` as input and returns logits during the forward pass.

# How To Start Training?

You need to create an `LLMTrainer` object and call `.train()` on it. Read about its parameters below: 

### `LLMTrainer()` parameters

```python
model:        torch.nn.Module = None,                      # The neural network model to train  
optimizer:    torch.optim.Optimizer = None,                # Optimizer responsible for updating model weights  
scheduler:    torch.optim.lr_scheduler.LRScheduler = None, # Learning rate scheduler for dynamic adjustment
tokenizer:    tiktoken.Encoding = None                     # Tokenizer for generating text (used if verbose > 0 during training)
model_returns_logits: bool = False                         # Whether model(X) returns logits or an object with an attribute `logits`
```

You must specify only the `model`. The other attributes are optional and will be set to default values if not specified.

### `LLMTrainer.train()` Parameters

| Parameter                 | Type               | Description                                                       | Default value           |
|---------------------------|--------------------|-------------------------------------------------------------------|-------------------------|
| `max_steps`               | `int`              | The maximum number of training steps                              | **5,000**               |
| `save_each_n_steps`       | `int`              | The interval of steps at which to save model checkpoints          | **1,000**               |
| `print_logs_each_n_steps` | `int`              | The interval of steps at which to print training logs             | **1**                   |
| `BATCH_SIZE`              | `int`              | The total batch size for training                                 | **256**                 |
| `MINI_BATCH_SIZE`         | `int`              | The mini-batch size for gradient accumulation                     | **16**                  |
| `context_window`          | `int`              | The context window size for the data loader                       | **128**                 |
| `data_dir`                | `str`              | The directory containing the training data                        | **"data"**              |
| `logging_file`            | `Union[str, None]` | The file path for logging training metrics                        | **"logs_training.csv"** |
| `generate_each_n_steps`   | `int`              | The interval of steps at which to generate and print text samples | **200**                 |
| `prompt`                  | `str`              | Beginning of the sentence that the model will continue            | **"Once upon a time"**  |
| `save_dir`                | `str`              | The directory to save model checkpoints                           | **"checkpoints"**       |


Every parameter has a default value, so you can start training simply by calling `LLMTrainer.train()`.

# To contribute

1. Fork the repository.
2. Make changes.
3. Apply linter.
```
$ pip install pylint==3.3.5
$ pylint $(git ls-files '*.py')
```
4. Commit and push your changes.
5. Create a pull request from your fork to the main repository.
