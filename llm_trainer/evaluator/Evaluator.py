from typing import Literal

import torch
from transformers import PreTrainedTokenizer, AutoTokenizer

from llm_trainer.evaluator.hellaswag import evaluate_hellaswag

class Evaluator:
    """
    A class for evaluating Large Language Models (LLMs) on various benchmarks.
    
    This class provides a unified interface for evaluating LLMs on different benchmarks.
    Currently supports the HellaSwag benchmark, with potential for additional benchmarks
    in the future.
    """

    def evaluate(self,
                 model: torch.nn.Module = None,
                 tokenizer: PreTrainedTokenizer | AutoTokenizer = None,
                 dataset: Literal["hellaswag"] = "hellaswag",
                 verbose: bool = True,
                 return_logits: bool = True) -> None:
        """
        Evaluates a provided LLM on a specified benchmark.
        
        Args:
            model (object): The LLM model to evaluate. This should be a model object that
                can perform forward passes and return logits or predictions.
            dataset (Literal["hellaswag"], optional): The benchmark dataset to evaluate on.
                Currently only supports "hellaswag". Defaults to "hellaswag".
            verbose (bool, optional): Whether to print detailed evaluation progress and results.
                Defaults to True.
            return_logits (bool, optional): Whether the model's forward pass returns raw logits
                or an object with a 'logits' attribute. Defaults to True.
            
        Raises:
            ValueError: If an unsupported benchmark dataset is specified.
        """
        match dataset:
            case "hellaswag":
                evaluate_hellaswag(model=model,
                                   tokenizer=tokenizer,
                                   verbose=verbose,
                                   return_logits=return_logits)
            case _:
                raise ValueError("Unknown benchmark.")
