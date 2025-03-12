import os
import json
from typing import Literal

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def create_dataset(save_dir: str = "data",
                   dataset: str = Literal["fineweb-edu-10B"],
                   chunks_limit: int = 1_500,
                   chunk_size=int(1e6),
                   tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2"),
                   end_of_text_token: int = 50256) -> None:
    """
    Creates a tokenized dataset from a Hugging Face dataset and stores it in chunks.

    Parameters:
        save_dir (str):
            Directory where tokenized chunks will be saved.
        dataset (str):
            Dataset to create. Supported datasets: ["fineweb-edu-10B"]
        chunks_limit(int):
            Maximum number of chunks to store.
        chunk_size (int):
            Number of tokens per chunk.
        tokenizer (tiktoken.Encoding):
            Which tokenizer to use to prepare a dataset.
        end_of_text_token (int):
            Token id of the end of text token.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset with streaming enabled to avoid high memory usage
    dataset = load_dataset(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
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
    chunk_tokens: np.ndarray = np.empty((chunk_size,), dtype=np.uint16)
    chunk_index: int = 0  # Track number of saved chunks
    n_chunk_tokens: int = 0  # Track current number of tokens in chunk
    
    # Initialize progress bar
    progress_bar = tqdm(total=chunks_limit, desc="Processing Chunks", unit="chunk")

    for tokens in (tokenize(doc) for doc in dataset):
        if chunk_index >= chunks_limit:
            break  # Stop if the chunk limit is reached
        
        if n_chunk_tokens + len(tokens) < chunk_size:
            # Add tokens to the current chunk
            chunk_tokens[n_chunk_tokens:n_chunk_tokens + len(tokens)] = tokens
            n_chunk_tokens += len(tokens)
        else:
            # Save the full chunk
            filename = os.path.join(save_dir, f"chunk_{chunk_index:04d}.npy")
            remaining_space = chunk_size - n_chunk_tokens
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


def create_dataset_from_json(save_dir: str = "data",
                             json_dir: str = "json_files",
                             chunks_limit: int = 1_500,
                             chunk_size=int(1e6),
                             tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2"),
                             eot_token: int = 50256) -> None:
    """
    Creates a tokenized dataset from JSON files containing text documents.
    Each JSON file in `json_dir` must contain a list of documents. Each document must have a key `text`.

    Example JSON file:
    ```
    [
        {"text": "Learn about LLMs: https://www.youtube.com/@_NickTech"},
        {"text": "Open-source python library to train LLMs: https://github.com/Skripkon/llm_trainer."},
        {"text": "My name is Nikolay Skripko. Hello from Russia (2025)."}
    ]
    ```

    Parameters:
        save_dir (str):
            Directory where tokenized chunks will be saved.
        json_dir (str):
            Directory containing JSON files with documents.
        chunks_limit (int):
            Maximum number of chunks to store.
        chunk_size (int):
            Number of tokens per chunk.
        tokenizer (tiktoken.Encoding):
            Tokenizer to encode text.
        eot_token (int):
            End-of-text token ID.
    """

    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]  # Create a list of JSON files from `json_dir`
    
    # Allocate space for chunk storage
    chunk_tokens: np.ndarray = np.empty((chunk_size,), dtype=np.uint16)
    chunk_index: int = 0  # Track number of saved chunks
    n_chunk_tokens: int = 0  # Track current number of tokens in chunk

    progress_bar = tqdm(total=chunks_limit, desc="Processing Chunks", unit="chunk")
    
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                documents = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {json_file}")
                continue
            
            for doc in documents:
                if "text" not in doc:
                    continue  # Skip if 'text' field is missing
                
                tokens = np.concatenate((tokenizer.encode_ordinary(doc["text"]), [eot_token])).astype(np.uint16)
                if chunk_index >= chunks_limit:
                    progress_bar.close()
                    return None

                if n_chunk_tokens + len(tokens) < chunk_size:
                    chunk_tokens[n_chunk_tokens:n_chunk_tokens + len(tokens)] = tokens
                    n_chunk_tokens += len(tokens)
                else:
                    filename = os.path.join(save_dir, f"chunk_{chunk_index:04d}.npy")
                    remaining_space = chunk_size - n_chunk_tokens
                    chunk_tokens[n_chunk_tokens:n_chunk_tokens + remaining_space] = tokens[:remaining_space]
                    np.save(filename, chunk_tokens)
                    chunk_index += 1
                    progress_bar.update(1)
                    
                    if chunk_index >= chunks_limit:
                        progress_bar.close()
                        return None
                    
                    chunk_tokens[:len(tokens) - remaining_space] = tokens[remaining_space:]
                    n_chunk_tokens = len(tokens) - remaining_space
    
    progress_bar.close()
    return None
