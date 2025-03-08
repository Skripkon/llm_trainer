from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMLMModel, xLSTMLMModelConfig

from llm_trainer import LLMTrainer, create_dataset

if __name__ == "__main__":
    
    # Step 1: Create a default dataset
    create_dataset(save_dir="data", dataset="fineweb-edu-10B", CHUNKS_LIMIT=100, CHUNK_SIZE=int(1e6))

    # Step 2: Define any LLM model (xLSTM in this example -- https://github.com/NX-AI/xlstm)
    conf = OmegaConf.load("xlstm_config.yaml")
    cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(conf), config=DaciteConfig(strict=True))
    xLSTM = xLSTMLMModel(cfg)

    # Step 3: Create an LLMTrainer object
    trainer = LLMTrainer(model=xLSTM,
                      optimizer=None,  # defaults to AdamW
                      scheduler=None,  # defaults to Warm-up steps + cosine annealing
                      tokenizer=None,  # defaults to GPT2 tokenizer
                      )

    # Step 4: Start training
    trainer.train(max_steps=1_000,
                  verbose=100,                      # Sample from the model each 50 steps
                  context_window=128,               # Context window of the model
                  data_dir="data",                  # Directory with .npy files containing tokens
                  BATCH_SIZE=256,                   # Batch size
                  MINI_BATCH_SIZE=16,               # Gradient accumulation is used. BATCH_SIZE = MINI_BATCH_SIZE * accumulation_steps
                  logging_file="logs_training.csv", # File to write logs of the training
                  save_each_n_steps=500,            # Save the state each 100 steps
                  save_dir="checkpoints"            # Directory where to save training state (model + optimizer + datalader)
    )
