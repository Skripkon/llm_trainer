"""
Downloads and evaluates HellaSwag in Python.
==========================================
- GPT2 (124M): 0.2955
- BERT-Base 110M: 40.5

Code is adapted from Andrej Karpathy's repository
"""

from datasets import load_dataset
import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizer, AutoTokenizer
from accelerate.test_utils.testing import get_backend


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

@torch.no_grad()
def evaluate_hellaswag(model: torch.nn.Module = None,
                       tokenizer: PreTrainedTokenizer | AutoTokenizer = None,
                       verbose: int = 1_000,
                       return_logits: bool = True):

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
            print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
        
    print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
