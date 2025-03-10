import os
from typing import Literal

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def create_dataset(save_dir: str = "data",
                   dataset: str = Literal["fineweb-edu-10B"],
                   CHUNKS_LIMIT: int = 1_500,
                   CHUNK_SIZE=int(1e6)):
    """
    Creates a tokenized dataset from a Hugging Face dataset and stores it in chunks.

    Parameters:
        save_dir (str): Directory where tokenized chunks will be saved.
        dataset (str): Dataset to create. Supported datasets: ["fineweb-edu-10B"]
        CHUNKS_LIMIT (int): Maximum number of chunks to store.
        CHUNK_SIZE (int): Number of tokens per chunk.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset with streaming enabled to avoid high memory usage
    dataset = load_dataset(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # Initialize tokenizer (GPT-2 encoding)
    tokenizer = tiktoken.get_encoding("gpt2")
    end_of_text_token = tokenizer._special_tokens['<|endoftext|>']  # End-of-text delimiter
    
    def tokenize(doc):
        """
        Tokenizes a document if it's in English.
        
        Parameters:
            doc (dict): A document containing 'text' and 'language' fields.
        
        Returns:
            np.ndarray: Tokenized representation of the document.
        """
        if doc["language"] != "en":
            return []
        return np.concatenate((tokenizer.encode_ordinary(doc["text"]), [end_of_text_token])).astype(np.uint16)
    
    # Allocate space for chunk storage
    chunk_tokens: np.ndarray = np.empty((CHUNK_SIZE,), dtype=np.uint16)
    chunk_index: int = 0  # Track number of saved chunks
    n_chunk_tokens: int = 0  # Track current number of tokens in chunk
    
    # Initialize progress bar
    progress_bar = tqdm(total=CHUNKS_LIMIT, desc="Processing Chunks", unit="chunk")

    for tokens in (tokenize(doc) for doc in dataset):
        if chunk_index >= CHUNKS_LIMIT:
            break  # Stop if the chunk limit is reached
        
        if n_chunk_tokens + len(tokens) < CHUNK_SIZE:
            # Add tokens to the current chunk
            chunk_tokens[n_chunk_tokens:n_chunk_tokens + len(tokens)] = tokens
            n_chunk_tokens += len(tokens)
        else:
            # Save the full chunk
            filename = os.path.join(save_dir, f"chunk_{chunk_index:04d}.npy")
            remaining_space = CHUNK_SIZE - n_chunk_tokens
            chunk_tokens[n_chunk_tokens:n_chunk_tokens + remaining_space] = tokens[:remaining_space]
            np.save(file=filename, arr=chunk_tokens)
            
            # Update progress bar
            chunk_index += 1
            progress_bar.update(1)

            # Add remaining tokens to the next chunk
            chunk_tokens[:len(tokens) - remaining_space] = tokens[remaining_space:]
            n_chunk_tokens = len(tokens) - remaining_space
    
    # Close the progress bar
    progress_bar.close()
