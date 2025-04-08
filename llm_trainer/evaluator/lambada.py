from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import load_dataset
from accelerate.test_utils.testing import get_backend
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_lambada(model: torch.nn.Module = None,
                     tokenizer: PreTrainedTokenizer | AutoTokenizer = None,
                     verbose: int = 500,
                     return_logits: bool = True,
                     max_length: int = 256,
                     stride: int = 128):

    test = load_dataset("lambada", split="test")
    encodings = tokenizer.encode("\n\n".join(test["text"]), return_tensors="pt")

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
            ignore_index=-100,  # We don’t want the log-likelihood for the tokens
                                # we’re just treating as context to be included in our loss,
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
       
        if iteration % verbose == 0:
            avg_nll = neg_log_likelihood_total / n_tokens  # average negative log-likelihood per token
            ppl = torch.exp(avg_nll).item()
            print(f"{iteration}/{seq_len // stride + 1} Perplexity: {ppl}")

        iteration += 1

    avg_nll = neg_log_likelihood_total / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll).item()
    print(f"Perplexity: {ppl}")
