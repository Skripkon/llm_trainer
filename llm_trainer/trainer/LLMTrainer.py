import csv
import os
import time
import math
import sys

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import functional as F
import tiktoken
from tiktoken import Encoding

from llm_trainer.dataset.DataLoader import DataLoader
            
class LLMTrainer:
    def __init__(self,
                 model: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 tokenizer: Encoding = None,
                 model_returns_logits: bool = False):

        """
        Initializes an LLMTrainer instance.

        Parameters:
            model (torch.nn.Module): 
                The neural network model to be trained. Must be specified.
            
            optimizer (torch.optim.Optimizer, optional): 
                The optimizer used for training. If not provided, AdamW with weight decay and fused optimization is used.
            
            scheduler (torch.optim.lr_scheduler.LRScheduler, optional): 
                The learning rate scheduler. If not provided, a cosine annealing scheduler with warmup steps is used.
            
            tokenizer (tiktoken.Encoding, optional): 
                The tokenizer used to encode and decode text. Defaults to GPT-2 tokenizer.
            
            model_returns_logits (bool, optional): 
                Whether the model returns raw logits (`logits = model(X)`) or an object containing logits
                (`logits = model(X).logits`). Defaults to False.
        
        Raises:
            ValueError: If no model is provided.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"Training on device: CUDA, {torch.cuda.get_device_name()}")
        else:
            print("Training on device: CPU")

        if optimizer is None:
            optimizer = self._configure_optimizer(weight_decay=0.1, learning_rate=5e-3, model=model)
        self.optimizer = optimizer

        if scheduler is None:
            scheduler = CosineAnnealingWithWarmUpStepsScheduler(optimizer=self.optimizer)

        self.scheduler = scheduler

        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")

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
        ------
        Parameters:
            max_steps (int):
                The maximum number of training steps.
            save_each_n_steps (int):
                The interval of steps at which to save model checkpoints.
            print_logs_each_n_steps(int):
                The interval of steps at which to print training logs.
            BATCH_SIZE (int):
                The total batch size for training.
            MINI_BATCH_SIZE (int):
                The mini-batch size for gradient accumulation.
            context_window (int):
                The context window size for the data loader.
            data_dir (str):
                The directory containing the training data.
            logging_file (str):
                The file path for logging training metrics.
            generate_each_n_steps (int):
                The interval of steps at which to generate and print text samples.
            prompt (str):
                Beginning of the sentence that the model will continue (during generation).
            save_dir (str):
                The directory to save model checkpoints.
            overwrite_logging_file (bool):
                If logging file should be overwritten in case it exists. True by default,
                set to False if you continue training.
        """
        
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
                self._save_checkpoint(step, self.train_loader)


    def _generate_text(self, prompt: str = "Once upon a time", n_return_sequences: int = 4, length: int = 32) -> None:
        """
        Samples from the model and prints `n_return_sequences` continuation of the `prompt`.
        """

        # Make sure the model is on the same device
        self.model.to(self.device)
        self.model.eval()

        tokens = torch.Tensor(self.tokenizer.encode(prompt)).type(torch.long)
        tokens = tokens.unsqueeze(0).repeat(n_return_sequences, 1)

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
            decoded = self.tokenizer.decode(tokens=tokens)
            print(f"=== sample {i} ===\n{decoded}")

    def _save_checkpoint(self, step: int, train_loader: DataLoader) -> None:

        checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'step': step,
                    'train_loader': train_loader
                    }
        torch.save(checkpoint, f"checkpoints/cp_{step}.pth")

    def load_checkpoint(self, checkpoint_path: str) -> None:

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # If the model was saved after running `torch.compile` then the names of its layers were changed.
        # Need to change it back.
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    
        self.model.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_loader: DataLoader = checkpoint["train_loader"]
        # print(f"Starting from chunk: {self.train_loader.current_chunk}")

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
