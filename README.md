# `llm_trainer` in 5 Lines of Code

```python
from llm_trainer import create_dataset, LLMTrainer

create_dataset(save_dir="data")   # Generate the default FineWeb dataset
model = ...                       # Define or load your model
trainer = LLMTrainer(model)       # Initialize trainer with default settings
trainer.train(data_dir="data")    # Start training on the dataset
```

# How to Prepare Data

1. Find a text corpus.
2. Tokenize the text.
3. Save the tokens as `.npy` files in a single folder (`your_data_dir`). You can save them as one file or multiple files.

   **Note:** The `.npy` format is the result of using `np.save(filename, tokens)`.

This process is handled by `llm_trainer.create_dataset()`, which currently supports only datasets from Hugging Face:

```python
def create_dataset(save_dir:     str = "data",
                   HF_path:      str = "HuggingFaceFW/fineweb-edu",
                   HF_name:      str = "sample-10BT",
                   CHUNKS_LIMIT: int = 1_500,    # Number of chunks (files) to create
                   CHUNK_SIZE:   int = int(1e6)) # Tokens per chunk (file)
```

4. Pass the data directory path to `LLMTrainer.train(data_dir="your_data_dir")`.

# Which Models Are Valid?

You can use **ANY** LLM that expects a tensor `X` with shape `(batch_size, context_window)` as input during the `.forward()` pass.

### `LLMTrainer` Attributes

```python
model:     torch.nn.Module = None,
optimizer: torch.optim.Optimizer = None,
scheduler: torch.optim.lr_scheduler.LRScheduler = None,
tokenizer: Encoding = None
```

You must specify only the `model`. The other attributes are optional and will be set to default values if not specified.

### `LLMTrainer` Parameters

```python
Parameters:
    max_steps (int, optional):
        The maximum number of training steps. Defaults to 5,000.
    verbose (int, optional):
        The interval of steps at which to generate and print text samples. Defaults to 200.
    save_each_n_steps (int, optional):
        The interval of steps at which to save model checkpoints. Defaults to 1,000.
    BATCH_SIZE (int, optional):
        The total batch size for training. Defaults to 256.
    MINI_BATCH_SIZE (int, optional):
        The mini-batch size for gradient accumulation. Defaults to 16.
    context_window (int, optional):
        The context window size for the data loader. Defaults to 128.
    data_dir (str, optional):
        The directory containing the training data. Defaults to "data".
    logging_file (Optional[str], optional):
        The file path for logging training metrics. Defaults to "logs_training.csv".
    save_dir (str, optional):
        The directory to save model checkpoints. Defaults to "checkpoints".
```

Every parameter has a default value, so you can start training simply by calling `LLMTrainer.train()`.

However, it is important to set the correct `context_window` size for your model.
