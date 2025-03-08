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
3. Save the tokens as `.npy` files in a single folder (`your_data_directory`). You can save them as one file or multiple files.

   **Note:** The `.npy` format is the result of using `np.save(filename, tokens)`.

This process is handled by `llm_trainer.create_dataset()`, which currently supports only one dataset from Hugging Face:

```python
def create_dataset(save_dir: str = "data",    # where to save created dataset
                   dataset: str = Literal["fineweb-edu-10B"],
                   CHUNKS_LIMIT: int = 1_500, # maximum number of files (chunks) with tokens to create
                   CHUNK_SIZE=int(1e6))       # number of tokens per chunk
```

4. Pass the data directory path to `LLMTrainer.train(data_dir="your_data_directory")`.

# Which Models Are Valid?

You can use **ANY** LLM that expects a tensor `X` with shape `(batch_size, context_window)` as input during the `.forward()` pass.

### `LLMTrainer` Attributes

```python
model:     torch.nn.Module = None,
optimizer: torch.optim.Optimizer = None,
scheduler: torch.optim.lr_scheduler.LRScheduler = None,
tokenizer: tiktoken.Encoding = None
```

You must specify only the `model`. The other attributes are optional and will be set to default values if not specified.

### `LLMTrainer` Parameters

| Parameter            | Type               | Description | Default value
|----------------------|--------------------|-------------------------------------------------------------------|-------------------------|
| `max_steps`          | `int`              | The maximum number of training steps                              | **5,000**               |
| `verbose`            | `int`              | The interval of steps at which to generate and print text samples | **200**                 |
| `save_each_n_steps`  | `int`              | The interval of steps at which to save model checkpoints          | **1,000**               |
| `BATCH_SIZE`         | `int`              | The total batch size for training                                 | **256**                 |
| `MINI_BATCH_SIZE`    | `int`              | The mini-batch size for gradient accumulation                     | **16**                  |
| `context_window`     | `int`              | The context window size for the data loader                       | **128**                 |
| `data_dir`           | `str`              | The directory containing the training data                        | **"data"**              |
| `logging_file`       | `Union[str, None]` | The file path for logging training metrics                        | **"logs_training.csv"** |
| `save_dir`           | `str`              | The directory to save model checkpoints                           | **"checkpoints"**       |



Every parameter has a default value, so you can start training simply by calling `LLMTrainer.train()`.

However, it is important to set the correct `context_window` size for your model.
