import os
import numpy as np
import torch

class DataLoader:
    def __init__(self, batch_size: int, context_window: int, data_dir: str = "data"):
        """
        Parameters:
            batch_size (int): Batch size
            context_window (int): Context window
            data_dir (str, optional): A directory with .npy files. Each file must contain tokens.
        """
        self.batch_size = batch_size
        self.context_window = context_window

        # Load the names of the chunks (there should be 1_500 files each of 1M tokens).
        chunks = os.listdir(data_dir)
        # Add "data/" before the files names
        chunks = [os.path.join(data_dir, chunk) for chunk in chunks]
        self.chunks = chunks
        self._shuffle_chunks()

        # Current_chunk is set to -1 because it will be incremented in the `_reset()` method.
        self.current_chunk: int = -1
        self._reset()

    def _reset(self):
        """
        Loads the tokens from the next chunk. If the chunk is the last in the dataset,
        it shuffles the chunks and starts from the first one.
        """
        self.current_chunk += 1

        if self.current_chunk == len(self.chunks) - 1:
            np.random.shuffle(self.chunks)
            self.current_chunk = 0

        self.tokens = self._load_tokens()
        self.current_position = 0

    def next_batch(self) -> tuple[torch.tensor, torch.tensor]:
        """
        Returns a batch of data.
        """
        B, T = self.batch_size, self.context_window
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        inputs = (buf[0:-1]).view(B, T)
        targets = (buf[1:]).view(B, T)
        self.current_position += B * T
    
        # if loading the next batch would be out of bounds, skip to next chunk
        if self.current_position + (B * T + 1) > len(self.tokens):
            self._reset()
        return inputs, targets

    def _load_tokens(self) -> torch.tensor:
        """
        Returns loaded tokens.
        """
        filename = self.chunks[self.current_chunk]
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt
    
    def _shuffle_chunks(self) -> None: 
        """
        Shuffles the chunks, because:
        1. FineWeb seems to have some internal structure.
        2. If you do more that one epoch (complete iteration over the whole dataset),
           model can learn some sequential patterns in the data. It's always a good idea to shuffle your training dataset before each new epoch.
        """
        np.random.shuffle(self.chunks)  # in-place
