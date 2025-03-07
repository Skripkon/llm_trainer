import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def create_dataset(save_dir: str = "data", HF_path: str = "HuggingFaceFW/fineweb-edu", HF_name: str = "sample-10BT", CHUNKS_LIMIT: int = 1_500):

    # Create a directory for the data if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Connect to the dataset (set streaming=True to disable downloading it because it will demand too much memory)
    fw = load_dataset(path=HF_path, name=HF_name, split="train", streaming=True)

    # Each document from this dataset is a dictionary.
    # doc["text"] = text that we want to encode and train our LLM on.
    # doc["language"] = language in which the text is written. For the sake of simplicity we'll keep only 'en' (English)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>'] # end of text token (used as a delimiter between documents)

    def tokenize(doc):
        """Tokenizes a single document and returns a numpy array of uint16 tokens"""
        return np.concatenate(
            (enc.encode_ordinary(doc["text"]), [eot])).astype(np.uint16)

    CHUNK_SIZE = int(1e6) # 1M tokens per chunk
    # Allocate space for the current chunk
    chunk_tokens: np.ndarray = np.empty((CHUNK_SIZE,), dtype=np.uint16)
    chunk_index: int = 0
    n_chunk_tokens: int = 0

    # 1. Tokenize texts while len(tokens) <= 1M tokens.
    # 2. Save a new chunk
    # 3. Small chunk size is used to shuffle the data by shuffling the chunks during the training.

    # For my purposes, 1.5B tokens will be enough.
    # 1.5B = 1M * 1500 => I need only 1.5k chunks

    # Initialize progress bar
    progress_bar = tqdm(total=CHUNKS_LIMIT, desc="Processing Chunks", unit="chunk")

    for tokens in (tokenize(doc) for doc in fw):
        if chunk_index >= CHUNKS_LIMIT:
            break  # Stop if the chunk limit is reached

        if n_chunk_tokens + len(tokens) < CHUNK_SIZE:
            chunk_tokens[n_chunk_tokens:n_chunk_tokens + len(tokens)] = tokens
            n_chunk_tokens += len(tokens)
        else:  # Finish and save the current chunk
            filename = os.path.join(save_dir, f"chunk_{chunk_index:04d}.npy")
            rem = CHUNK_SIZE - n_chunk_tokens
            chunk_tokens[n_chunk_tokens:n_chunk_tokens + rem] = tokens[:rem]
            np.save(file=filename, arr=chunk_tokens)

            # Update progress bar
            chunk_index += 1
            progress_bar.update(1)

            # There are tokens that weren't fit into the current chunk. Add them to the next chunk
            chunk_tokens[:len(tokens) - rem] = tokens[rem:]
            n_chunk_tokens = len(tokens) - rem

    # Close the progress bar
    progress_bar.close()
