import csv
import os
import time
import math
import sys

from accelerate.test_utils.testing import get_backend
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import functional as F
from transformers import PreTrainedTokenizer, AutoTokenizer

from llm_trainer.dataset.DataLoader import DataLoader

class LLMTrainer:
    """
    A trainer class for fine-tuning and training Large Language Models (LLMs).
    
    This class provides a comprehensive training pipeline for LLMs, including:
    - Gradient accumulation for handling large batch sizes
    - Automatic mixed precision training
    - Learning rate scheduling with warmup
    - Checkpointing and resuming training
    - Text generation during training
    - Training metrics logging
    
    The trainer supports any PyTorch model that takes input tensors of shape (batch_size, context_window)
    and returns logits during the forward pass.
    
    Example:
        >>> model = YourLLMModel()
        >>> trainer = LLMTrainer(model=model)
        >>> trainer.train()
    """

    def __init__(self,
                 model: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 tokenizer: PreTrainedTokenizer | AutoTokenizer = None,
                 model_returns_logits: bool = False):

        """
        Initializes an LLMTrainer instance.
        ------
        Parameters:
            model (torch.nn.Module): 
                The neural network model to be trained. Must be specified.
            
            optimizer (torch.optim.Optimizer): 
                The optimizer used for training. If not provided, AdamW with weight decay and fused optimization is used.
            
            scheduler (torch.optim.lr_scheduler.LRScheduler): 
                The learning rate scheduler. If not provided, a cosine annealing scheduler with warmup steps is used.
            
            tokenizer (transformers.PreTrainedTokenizer): 
                The tokenizer used to encode and decode text. Defaults to GPT-2 tokenizer.
            
            model_returns_logits (bool): 
                Whether the model returns raw logits (`logits = model(X)`) or an object containing logits
                (`logits = model(X).logits`). Defaults to False.
        
        Raises:
            ValueError: If no model is provided.
        """

        self.device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        print(f"Training on: {self.device}")

        if optimizer is None:
            optimizer = self._configure_optimizer(weight_decay=0.1, learning_rate=5e-3, model=model)
        self.optimizer = optimizer

        if scheduler is None:
            scheduler = CosineAnnealingWithWarmUpStepsScheduler(optimizer=self.optimizer)

        self.scheduler = scheduler

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        self.tokenizer = tokenizer

        if model is None:
            raise ValueError("Specify a model.")
        self.model = model

        self.train_loader = None
        self.current_step: int = 0

        self.model_returns_logits = model_returns_logits

    def train(self,
              max_steps: int = 5_000,
              save_each_n_steps: int = 1000,
              print_logs_each_n_steps: int = 1,
              BATCH_SIZE: int = 256,
              MINI_BATCH_SIZE: int = 16,
              context_window: int = 256,
              data_dir: str = "data",
              logging_file: str = "logs_training.csv",
              generate_each_n_steps: int = 250,
              prompt: str = "Once upon a time",
              save_dir: str = "checkpoints",
              overwrite_logging_file: bool = True) -> None:
        """
        Train the model with the specified parameters.
        
        This method implements the main training loop with the following features:
        - Gradient accumulation to handle large effective batch sizes
        - Automatic mixed precision training using bfloat16
        - Learning rate scheduling with warmup
        - Regular checkpointing and logging
        - Text generation samples during training
        - Performance metrics tracking (tokens/sec, loss, etc.)
        
        Parameters:
            max_steps (int):
                The maximum number of training steps to perform.
            save_each_n_steps (int):
                The interval of steps at which to save model checkpoints.
            print_logs_each_n_steps (int):
                The interval of steps at which to print training logs to console.
            BATCH_SIZE (int):
                The total effective batch size for training. This is achieved through
                gradient accumulation over multiple mini-batches.
            MINI_BATCH_SIZE (int):
                The actual mini-batch size processed in each forward pass. Must be
                a divisor of BATCH_SIZE.
            context_window (int):
                The context window size for the data loader. This determines the
                maximum sequence length the model can process.
            data_dir (str):
                The directory containing the training data in .npy format.
            logging_file (str):
                The file path for logging training metrics in CSV format.
            generate_each_n_steps (int):
                The interval of steps at which to generate and print text samples
                using the current model state.
            prompt (str):
                The initial text prompt used for generation during training.
            save_dir (str):
                The directory where model checkpoints will be saved.
            overwrite_logging_file (bool):
                Whether to overwrite the logging file if it exists. Set to False
                when continuing training from a checkpoint.
                
        Note:
            The training process uses gradient accumulation to achieve the effective
            BATCH_SIZE. The number of accumulation steps is calculated as
            BATCH_SIZE // MINI_BATCH_SIZE.
            
        Example:
            >>> trainer = LLMTrainer(model=model)
            >>> trainer.train(
            ...     max_steps=10000,
            ...     BATCH_SIZE=512,
            ...     MINI_BATCH_SIZE=32,
            ...     context_window=1024
            ... )
        """

        # Sets the internal precision of float32 matrix multiplications.
        torch.set_float32_matmul_precision('high')

        # Make sure that a directory for checkpoints exists
        os.makedirs(name=save_dir, exist_ok=True)

        if self.train_loader is None:
            self.train_loader = DataLoader(batch_size=MINI_BATCH_SIZE, context_window=context_window, data_dir=data_dir)

        if overwrite_logging_file or not os.path.exists(logging_file):
            # Create a file for training logs and add header to it
            with open(logging_file, mode="w", newline="", encoding="utf8") as file:
                writer = csv.writer(file)
                writer.writerow(["Step", "Loss", "Norm", "LR", "dt (ms)", "Tokens/sec"])

        gradient_accumulation_steps: int = BATCH_SIZE // MINI_BATCH_SIZE
        
        self.model.train()
        self.model.to(self.device)

        # torch.compile requires Triton (https://github.com/triton-lang/triton), which is only supported on Linux.
        if sys.platform in {"linux", "linux2"}:
            self.model = torch.compile(self.model)

        for step in range(self.current_step, max_steps):
            t0 = time.time()
            last_step = (step == max_steps - 1)
            self.optimizer.zero_grad()

            # Gradient accumulation is applied to maintain a bigger batch_size
            loss_accum = 0
            for _ in range(gradient_accumulation_steps):

                inputs, targets = self.train_loader.next_batch()
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Use lower precision for higher bandwidth.
                # Don't use torch.float16 because it will require gradient rescaling (since float16 represents a limited range)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    if self.model_returns_logits:
                        logits = self.model(inputs)
                    else:
                        logits = self.model(inputs).logits

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
                loss_accum += loss.detach()

            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            # LOGGING
            t1 = time.time()
            dt = t1 - t0  # time elapsed in seconds
            tokens_processed = gradient_accumulation_steps * self.train_loader.batch_size * self.train_loader.context_window
            tokens_per_sec = tokens_processed / dt

            # Open the CSV file in append mode
            with open(logging_file, mode="a", newline="", encoding="utf8") as file:
                writer = csv.writer(file)
                writer.writerow([step, f"{loss_accum:.6f}", f"{norm:.4f}", f"{self.scheduler.get_last_lr()[0]:.4e}", f"{dt * 1000:.2f}", f"{tokens_per_sec:.2f}"])

            if step % print_logs_each_n_steps == 0:
                print(f"step: {step} | Loss: {loss_accum:.6f} | norm: {norm:.4f} | lr: {self.scheduler.get_last_lr()[0]:.6e} | dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}")

            # Sample from the model
            if ((step > 0 and step % generate_each_n_steps == 0) or last_step):
                self._generate_text(prompt=prompt)
                self.model.train()  # during generation model is set to .eval() mode

            # Save the model (checkpoint)
            if last_step or ((step > 0) and ((step % save_each_n_steps) == 0)):
                self._save_checkpoint(step, self.train_loader, save_dir)


    def _generate_text(self, prompt: str = "Once upon a time", n_return_sequences: int = 4, length: int = 32) -> None:
        """
        Generate text samples from the model using top-k sampling.
        
        This method generates multiple continuations of the given prompt using the current
        model state. It uses top-k sampling with k=10 for text generation, which helps
        maintain diversity.
        
        Parameters:
            prompt (str):
                The initial text prompt to continue. Defaults to "Once upon a time".
            n_return_sequences (int):
                The number of different continuations to generate. Defaults to 4.
            length (int):
                The total length of each generated sequence (including the prompt).
                Defaults to 32 tokens.
                
        Note:
            The model is temporarily set to evaluation mode during generation
            and returned to training mode afterward.
            
        Example:
            >>> trainer._generate_text(
            ...     prompt="The quick brown fox",
            ...     n_return_sequences=2,
            ...     length=50
            ... )
            === sample 0 ===
            The quick brown fox jumps over the lazy dog...
            === sample 1 ===
            The quick brown fox runs through the forest...
        """

        # Make sure the model is on the same device
        self.model.to(self.device)
        self.model.eval()

        tokens = self.tokenizer.encode(prompt, return_tensors="pt").type(torch.long)
        tokens = tokens.repeat(n_return_sequences, 1)

        generated_tokens = tokens.to(self.device)
        with torch.no_grad():
            while generated_tokens.size(1) < length:

                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    if self.model_returns_logits:
                        logits = self.model(generated_tokens)
                    else:
                        logits = self.model(generated_tokens).logits

                # logits.shape = (batch_size, context_window, vocab_size)

                logits = logits[:, -1, :]  # Get last token logits (B, vocab_size)
                probs = F.softmax(logits, dim=-1)  # Convert to probabilities

                # Top-k sampling
                topk_probs, topk_indices = torch.topk(probs, k=10, dim=-1)
                sampled_indices = torch.multinomial(topk_probs, 1)  # Shape: (B, 1)
                next_tokens = torch.gather(topk_indices, -1, sampled_indices)  # (B, 1)

                # Append generated token to sequence
                generated_tokens = torch.cat((generated_tokens, next_tokens), dim=1)

        # print the generated text
        for i in range(n_return_sequences):
            tokens = generated_tokens[i, :length].tolist()
            decoded = self.tokenizer.decode(tokens)
            print(f"=== sample {i} ===\n{decoded}")

    def _save_checkpoint(self, step: int, train_loader: DataLoader, save_dir: str = "checkpoints") -> None:

        checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'step': step,
                    'train_loader': train_loader
                    }
        torch.save(checkpoint, f"{save_dir}/cp_{step}.pth")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint and restore the training state.
        
        This method loads a previously saved checkpoint and restores:
        - The model's state dictionary
        - The optimizer's state dictionary
        - The training data loader
        - The current training step
        
        Parameters:
            checkpoint_path (str):
                Path to the checkpoint file (.pth) to load.
                
        Note:
            If the model was saved after running `torch.compile`, the method
            automatically handles the layer name changes by removing the
            "_orig_mod." prefix from the state dictionary keys.
            
        Example:
            >>> trainer = LLMTrainer(model=model)
            >>> trainer.load_checkpoint("checkpoints/cp_1000.pth")
            >>> trainer.train()  # Continues training from step 1000
        """

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # If the model was saved after running `torch.compile` then the names of its layers were changed.
        # Need to change it back.
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        self.model.to(self.device)
        self.model.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_loader: DataLoader = checkpoint["train_loader"]

        self.current_step = checkpoint['step']  # Resume from the last step

    def _configure_optimizer(self, weight_decay, learning_rate, model):
        # Consider all the parameters.
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

        # Weight decay only 2D parameters.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Create AdamW optimizer (fused version requires CUDA + runs only on Linux)
        use_fused = (self.device == self.device) and (sys.platform in {"linux", "linux2"})
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-5, fused=use_fused)
        return optimizer

    def plot_loss(self, logging_file: str = "logs_training.csv", smoothing_window: int = 10):
        """
        Visualize the training loss over time with optional smoothing.
        
        This method creates a plot showing both the raw and smoothed training loss
        curves. The smoothed curve helps visualize the overall trend by reducing
        noise. The plot also includes horizontal reference lines at common loss
        values to help assess model performance.
        
        Parameters:
            logging_file (str):
                Path to the CSV file containing training logs. Defaults to
                "logs_training.csv".
            smoothing_window (int):
                Size of the rolling window used for smoothing the loss curve.
                Larger values result in smoother curves but may obscure short-term
                trends. Defaults to 10.
                
        Note:
            The plot includes reference lines at loss values of 3, 4, 5, and 6
            to help gauge model performance.
            
        Example:
            >>> trainer.plot_loss(
            ...     logging_file="training_logs.csv",
            ...     smoothing_window=20
            ... )
        """

        data = pd.read_csv(logging_file)
        smoothed_loss = data["Loss"].rolling(window=smoothing_window).mean()

        plt.plot(data["Step"], smoothed_loss, label="Smoothed Loss", color="pink")
        plt.plot(data["Step"], data["Loss"], alpha=0.5, label="Original Loss", color="gray")

        plt.axhline(y=6, color='r', linestyle='--', alpha=0.6)
        plt.axhline(y=5, color='gray', linestyle='--', alpha=0.6)
        plt.axhline(y=4, color='y', linestyle='--', alpha=0.6)
        plt.axhline(y=3, color='g', linestyle='--', alpha=0.6)

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


class CosineAnnealingWithWarmUpStepsScheduler(LRScheduler):
    """
    Default scheduler.
    """

    def __init__(self, optimizer: torch.optim.Optimizer = None, min_lr: float = 1e-4, max_lr: float = 5e-3, warm_up_steps: int = 750, max_steps: int = 5000):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warm_up_steps = warm_up_steps
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):

        # Warmup phase
        if self._step_count < self.warm_up_steps:
            return [self.max_lr * (self._step_count + 1) / self.warm_up_steps]

        # Late phase
        if self._step_count > self.max_steps:
            return [self.min_lr]
        
        # Cosine annealing phase
        decay_ratio = (self._step_count - self.warm_up_steps) / (self.max_steps - self.warm_up_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (self.max_lr - self.min_lr)]
