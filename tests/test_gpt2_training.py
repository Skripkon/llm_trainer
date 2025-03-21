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
        n_positions=128,
        n_embd=32,
        n_layer=4,
        n_head=4,
    )

    gpt2_model = GPT2LMHeadModel(gpt2_config)
    trainer = LLMTrainer(model=gpt2_model)

    trainer.train(max_steps=10,
                  generate_each_n_steps=3,
                  print_logs_each_n_steps=1,
                  context_window=128,
                  data_dir="data",
                  BATCH_SIZE=32,
                  MINI_BATCH_SIZE=16,
                  logging_file="logs_training.csv",
                  save_each_n_steps=1_000,
                  save_dir="checkpoints",
                  prompt="Once upon a time in Russia"
    )
