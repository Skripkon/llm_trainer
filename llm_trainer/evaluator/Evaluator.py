import logging
from typing import TypedDict

from accelerate.test_utils.testing import get_backend
from datasets import load_dataset
from tabulate import tabulate
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, AutoTokenizer


class DatasetConfig(TypedDict):
    name: str
    config_name: str | None
    text_key: str  # The key in the dataset that contains the text to evaluate

class Evaluator:
    """
    A class for evaluating Large Language Models (LLMs) on various benchmarks.

    Methods:
        eval_hellaswag()
        eval_lambada()
        eval_wikitext_103()
        eval_wikitext_2()
        eval_wikitext_ptb()

        eval_all()
    """

    # Dataset configurations
    LAMBADA_CONFIG: DatasetConfig = {
        "name": "lambada",
        "config_name": None,
        "text_key": "text"
    }

    WIKITEXT_103_CONFIG: DatasetConfig = {
        "name": "wikitext",
        "config_name": "wikitext-103-raw-v1",
        "text_key": "text"
    }

    WIKITEXT_2_CONFIG: DatasetConfig = {
        "name": "wikitext",
        "config_name": "wikitext-2-raw-v1",
        "text_key": "text"
    }

    PTB_CONFIG: DatasetConfig = {
        "name": "ptb_text_only",
        "config_name": None,
        "text_key": "sentence"
    }

    @torch.no_grad()
    def eval_hellaswag(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer | AutoTokenizer,
                       verbose: int = 1_000, return_logits: bool = False) -> float:

        """
        Evaluates a provided LLM on the HellaSwag benchmark.
        
        Args:
            model (torch.nn.Module): The LLM model to evaluate. Must be capable of autoregressive
                language modeling.
            tokenizer (PreTrainedTokenizer | AutoTokenizer): The tokenizer compatible with the model.
            verbose (int): Frequency of progress reporting. Defaults to 1,000.
            return_logits (bool): Whether the model returns raw logits or a wrapper containing logits.
                Defaults to True.

        Returns:
            Accuracy.
        """

        def render_example(example, tokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
            """
            Given the example as a dictionary, render it as three torch tensors:
            - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
            - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
            - label (the index of the correct completion, which we want to have the highest likelihood)
            """
            label = int(example["label"])
            endings = example["endings"]

            # Gather up all the tokens.
            ctx_tokens = tokenizer.encode(example["ctx"])
            tok_rows = []
            mask_rows = []
            for end in endings:
                end_tokens = tokenizer.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
                tok_rows.append(ctx_tokens + end_tokens)
                mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

            # Note, that the number of tokens in each row can differ.
            max_len = max(len(row) for row in tok_rows)
            tokens = torch.zeros((4, max_len), dtype=torch.long)
            mask = torch.zeros((4, max_len), dtype=torch.long)
            for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
                tokens[i, :len(tok_row)] = torch.tensor(tok_row)
                mask[i, :len(mask_row)] = torch.tensor(mask_row)

            return tokens, mask, label

        device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        model.to(device)

        torch.set_float32_matmul_precision('high') # use tf32
        num_correct_norm = 0
        num_correct = 0
        num_total = 0
        dataset = load_dataset("hellaswag")

        for example in dataset["validation"]:
            tokens, mask, label = render_example(example, tokenizer)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # get the logits
            if return_logits:
                logits = model(tokens)
            else:
                logits = model(tokens).logits

            # evaluate the autoregressive loss at all positions
            shift_logits = (logits[..., :-1, :]).contiguous()
            shift_tokens = (tokens[..., 1:]).contiguous()
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_tokens = shift_tokens.view(-1)
            shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
            shift_losses = shift_losses.view(tokens.size(0), -1)
            # now get the average loss just for the completion region (where mask == 1), in each row
            shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
            masked_shift_losses = shift_losses * shift_mask
            # sum and divide by the number of 1s in the mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            # now we have a loss for each of the 4 completions
            # the one with the lowest loss should be the most likely
            pred = sum_loss.argmin().item()
            pred_norm = avg_loss.argmin().item()

            # accumulate stats
            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)
            if verbose > 0 and num_total % verbose == 0:
                print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm / num_total:.4f}")

        if verbose > 0:
            print(f"{num_total}/{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm / num_total:.4f}")
        
        return num_correct_norm / num_total


    @torch.no_grad()
    def eval_lambada(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer | AutoTokenizer,
                     verbose: int = 100, return_logits: bool = False, max_length: int = 1024, stride: int = 512) -> float:
        return self._compute_perplexity(
            hf_dataset=self.LAMBADA_CONFIG["name"],
            config_name=self.LAMBADA_CONFIG["config_name"],
            text_key=self.LAMBADA_CONFIG["text_key"],
            model=model,
            tokenizer=tokenizer,
            verbose=verbose,
            return_logits=return_logits,
            max_length=max_length,
            stride=stride
        )

    @torch.no_grad()
    def eval_wikitext_103(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer | AutoTokenizer, 
                         verbose: int = 100, return_logits: bool = False, max_length: int = 1024, stride: int = 512) -> float:
        return self._compute_perplexity(
            hf_dataset=self.WIKITEXT_103_CONFIG["name"],
            config_name=self.WIKITEXT_103_CONFIG["config_name"],
            text_key=self.WIKITEXT_103_CONFIG["text_key"],
            model=model,
            tokenizer=tokenizer,
            verbose=verbose,
            return_logits=return_logits,
            max_length=max_length,
            stride=stride
        )
        
    @torch.no_grad()
    def eval_wikitext_2(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer | AutoTokenizer, 
                         verbose: int = 100, return_logits: bool = False, max_length: int = 1024, stride: int = 512) -> float:
        return self._compute_perplexity(
            hf_dataset=self.WIKITEXT_2_CONFIG["name"],
            config_name=self.WIKITEXT_2_CONFIG["config_name"],
            text_key=self.WIKITEXT_2_CONFIG["text_key"],
            model=model,
            tokenizer=tokenizer,
            verbose=verbose,
            return_logits=return_logits,
            max_length=max_length,
            stride=stride
        )

    @torch.no_grad()
    def eval_ptb(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer | AutoTokenizer, 
                 verbose: int = 100, return_logits: bool = False, max_length: int = 1024, stride: int = 512) -> float:
        return self._compute_perplexity(
            hf_dataset=self.PTB_CONFIG["name"],
            config_name=self.PTB_CONFIG["config_name"],
            text_key=self.PTB_CONFIG["text_key"],
            model=model,
            tokenizer=tokenizer,
            verbose=verbose,
            return_logits=return_logits,
            max_length=max_length,
            stride=stride
        )

    @torch.no_grad()
    def _compute_perplexity(self, hf_dataset: str, config_name: str | None, text_key: str,
                           model: torch.nn.Module, tokenizer: PreTrainedTokenizer | AutoTokenizer,
                           verbose: int = 100, return_logits: bool = False,
                           max_length: int = 1024, stride: int = 512) -> float:
        """
        Evaluates a provided LLM by computing perplexity on the specified benchmark.

        Args:
            hf_dataset (str): The name of the HuggingFace dataset
            config_name (Optional[str]): The configuration name for the dataset
            text_key (str): The key in the dataset that contains the text to evaluate
            model (torch.nn.Module): The LLM model to evaluate
            tokenizer (PreTrainedTokenizer | AutoTokenizer): The tokenizer compatible with the model
            verbose (int): Frequency of progress reporting
            return_logits (bool): Whether the model returns raw logits or a wrapper containing logits
            max_length (int): Maximum input sequence length for evaluation
            stride (int): The stride used when splitting text into overlapping chunks
        """
        test = load_dataset(hf_dataset, config_name, split="test")
        text = "\n\n".join(test[text_key])
    
        # Suppres the warning: Token indices sequence length is longer than the specified maximum sequence length for this model
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        encodings = tokenizer.encode(text, return_tensors="pt")
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)

        device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        model = model.to(device)
        seq_len = encodings.shape[1]

        neg_log_likelihood_total: float = 0.0
        n_tokens: int = 0
        prev_context_end: int = 0

        iteration: int = 1
        for context_start in range(0, seq_len, stride):
            context_end = min(context_start + max_length, seq_len)
            trg_len = context_end - prev_context_end

            input_ids = encodings[:, context_start:context_end].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                if return_logits:
                    logits = model(input_ids)
                else:
                    logits = model(input_ids).logits
            # logits.shape = (batch_size, seq_len, vocab_size)

            # Shift logits to compute loss
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_targets = target_ids[:, 1:].contiguous()

            # Compute loss
            neg_log_likelihood = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_targets.view(-1),
                ignore_index=-100,  # We don't want the log-likelihood for the tokens
                                    # we're just treating as context to be included in our loss,
                                    # so we can set these targets to -100 so that they are ignored 
                reduction='mean'
            )

            # Accumulate the total negative log-likelihood and the total number of tokens
            num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids

            num_loss_tokens = num_valid_tokens - 1  # subtract 1 due to label shift
            neg_log_likelihood_total += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_context_end = context_end
            if context_end == seq_len:
                break
        
            if verbose > 0 and iteration % verbose == 0:
                avg_nll = neg_log_likelihood_total / n_tokens  # average negative log-likelihood per token
                ppl = torch.exp(avg_nll).item()
                print(f"{iteration}/{seq_len // stride + 1} Perplexity: {ppl}")

            iteration += 1

        avg_nll = neg_log_likelihood_total / n_tokens  # average negative log-likelihood per token
        ppl = torch.exp(avg_nll).item()

        if verbose > 0:
            print(f"{iteration}/{iteration} Perplexity: {ppl}")

        return ppl

    @torch.no_grad()
    def eval_all(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer | AutoTokenizer,
                return_logits: bool = False, max_length: int = 1024, stride: int = 512):
        """
        Evaluates the model on all available benchmarks and prints a summary table.
        
        Args:
            model (torch.nn.Module): The LLM model to evaluate
            tokenizer (PreTrainedTokenizer | AutoTokenizer): The tokenizer compatible with the model
            return_logits (bool): Whether the model returns raw logits or a wrapper containing logits
            max_length (int): Maximum input sequence length for evaluation
            stride (int): The stride used when splitting text into overlapping chunks
        """
        
        results = []

        # Run HellaSwag evaluation
        print("\nEvaluating on HellaSwag...")
        acc = self.eval_hellaswag(model, tokenizer, 0, return_logits)
        results.append(["HellaSwag", "Accuracy", f"{acc:.4f}"])

        # Run LAMBADA evaluation
        print("\nEvaluating on LAMBADA...")
        ppl = self.eval_lambada(
            model=model,
            tokenizer=tokenizer,
            verbose=0,
            return_logits=return_logits,
            max_length=max_length,
            stride=stride
        )
        results.append(["LAMBADA", "Perplexity", f"{ppl:.2f}"])

        # Run WikiText-103 evaluation
        print("\nEvaluating on WikiText-103...")
        ppl = self.eval_wikitext_103(
            model=model,
            tokenizer=tokenizer,
            verbose=0,
            return_logits=return_logits,
            max_length=max_length,
            stride=stride
        )
        results.append(["WikiText-103", "Perplexity", f"{ppl:.2f}"])

        # Run WikiText-2 evaluation
        print("\nEvaluating on WikiText-2...")
        ppl = self.eval_wikitext_2(
            model=model,
            tokenizer=tokenizer,
            verbose=0,
            return_logits=return_logits,
            max_length=max_length,
            stride=stride
        )
        results.append(["WikiText-2", "Perplexity", f"{ppl:.2f}"])

        # Run PTB evaluation
        print("\nEvaluating on PTB...")
        ppl = self.eval_ptb(
            model=model,
            tokenizer=tokenizer,
            verbose=0,
            return_logits=return_logits,
            max_length=max_length,
            stride=stride
        )
        results.append(["PTB", "Perplexity", f"{ppl:.2f}"])
        
        # Print summary table
        print("\nEvaluation Summary:")
        print(tabulate(results, headers=["Benchmark", "Metric", "Value"], tablefmt="fancy_grid"))
