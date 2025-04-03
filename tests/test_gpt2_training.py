from transformers import GPT2LMHeadModel, GPT2Config

from llm_trainer import create_dataset 
from llm_trainer import LLMTrainer

def test_gpt2_training():

    create_dataset(save_dir="data",
                   dataset="fineweb-edu-10B",
                   chunks_limit=5,
                   chunk_size=int(1e6))

    gpt2_config = GPT2Config(
        vocab_size=50257,
        n_positions=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )

    gpt2_model = GPT2LMHeadModel(gpt2_config)
    trainer = LLMTrainer(model=gpt2_model)

    trainer.train(max_steps=5,
                  generate_each_n_steps=3,
                  print_logs_each_n_steps=1,
                  context_window=64,
                  data_dir="data",
                  BATCH_SIZE=16,
                  MINI_BATCH_SIZE=8,
                  logging_file="logs_training.csv",
                  save_each_n_steps=1_000,
                  save_dir="checkpoints",
                  prompt="Once upon a time in Russia"
    )
