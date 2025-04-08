from typing import Literal

import torch
from transformers import PreTrainedTokenizer, AutoTokenizer

from llm_trainer.evaluator.hellaswag import evaluate_hellaswag
from llm_trainer.evaluator.lambada import evaluate_lambada

class Evaluator:
    """
    A class for evaluating Large Language Models (LLMs) on various benchmarks.
    
    This class provides a unified interface for evaluating LLMs on different benchmarks.
    Currently supports the HellaSwag benchmark, with potential for additional benchmarks
    in the future.
    """

    def eval_hellaswag(self,
                 model: torch.nn.Module = None,
                 tokenizer: PreTrainedTokenizer | AutoTokenizer = None,
                 verbose: int = 1000,
                 return_logits: bool = True) -> None:
        """
        Evaluates a provided LLM on the HellaSwag benchmark.
        
        Args:
            model (torch.nn.Module): The LLM model to evaluate. Must be capable of autoregressive
                language modeling.
            tokenizer (PreTrainedTokenizer | AutoTokenizer): The tokenizer compatible with the model.
            verbose (int): Frequency of progress reporting. Defaults to 1,000.
            return_logits (bool): Whether the model returns raw logits or a wrapper containing logits.
                Defaults to True.
        """
        evaluate_hellaswag(model=model, tokenizer=tokenizer, verbose=verbose, return_logits=return_logits)

    def eval_lambada(self,
                 model: torch.nn.Module = None,
                 tokenizer: PreTrainedTokenizer | AutoTokenizer = None,
                 verbose: int = 1000,
                 return_logits: bool = True,
                 max_length: int = 1024,
                 stride: int = 512) -> None:
        """
        Evaluates a provided LLM on the LAMBADA benchmark.

        Args:
            model (torch.nn.Module): The LLM model to evaluate. Must be capable of autoregressive
                language modeling.
            tokenizer (PreTrainedTokenizer | AutoTokenizer): The tokenizer compatible with the model.
            verbose (int): Frequency of progress reporting. Defaults to 1,000.
            return_logits (bool): Whether the model returns raw logits or a wrapper containing logits.
                Defaults to True.
            max_length (int): Maximum input sequence length for evaluation. Defaults to 1024.
            stride (int): The stride used when splitting text into overlapping chunks. Helps with
                handling long contexts. Defaults to 512.
        """
        evaluate_lambada(model=model, tokenizer=tokenizer, verbose=verbose, return_logits=return_logits, max_length=max_length, stride=stride)
