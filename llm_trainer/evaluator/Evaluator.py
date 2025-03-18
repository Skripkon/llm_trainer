from typing import Literal

from llm_trainer.evaluator.hellaswag import evaluate_hellaswag

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, model, dataset: Literal["hellaswag"]):
        match dataset:
            case "hellaswag":
                evaluate_hellaswag(model)
            case _:
                raise ValueError("Unknown benchmark.")
