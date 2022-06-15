from dataclasses import dataclass
from typing import Union, Dict, List
from collections import Counter, defaultdict
from torch import Tensor

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
from terminaltables import github_table

def compute_scores(trues, preds):
    def format_score(score):
        return round(float(score), 4)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="macro", zero_division=0)
    b = format_score(balanced_accuracy_score(trues, preds))
    p = format_score(p)
    r = format_score(r)
    a = format_score(accuracy_score(trues, preds))

    return {'acc': a, 'pre': p, 'rec': r, 'sup': len(trues),
            'bal_acc': b, 'f1': format_score(f1)}


@dataclass
class ScoreWatcher:
    """Class for keeping track of an item in inventory."""
    score: Union[Tensor, float, int]
    mode: str = "min"
    steps: int = 0
    patience: int = 5
    delta: float = 0.05
    factor: float = 0.6
    min_weight: float = .05
    main: bool = False

    def update_steps_on_mode(self, score, weight: float, task: str):
        if self.main:  # Do not update weight on main !
            return self.steps, weight
        if (self.score - score) > self.delta:
            self.steps = 0
            self.score = score
        else:
            self.steps += 1
            if self.steps > self.patience:
                weight = self.factor * weight
                if weight < self.min_weight:
                    weight = self.min_weight
                    print(f"Minimal weight reached")
                else:
                    print(f"Weights have been updated for task `{task}` (Score: `{self.score}`, Weight: `{weight}` )")
                self.steps = 0
        return self.steps, weight

    def repr(self, task_name: str, weight: float):
        return f'<ScoreWatcher task="{task_name}" weight="{weight:.2f}" ' \
               f'best="{self.score:.2f}" mode="{self.mode}" />'


def get_confusion_matrix(trues: List[str], preds: List[str]):
    errors = defaultdict(Counter)
    for true, pred in zip(trues, preds):
        if true != pred:
            errors[true][pred] += 1

    return errors


def get_confusion_matrix_table(trues: List[str], preds: List[str]) -> List[List[str]]:
    """
    Returns a table formated confusion matrix
    """
    matrix = get_confusion_matrix(trues, preds)
    table = []
    # Retrieve each true prediction and its dictionary of errors
    for expected, pred_counter in matrix.items():
        counts = [(word, counter) for word, counter in sorted(
            pred_counter.items(), key=lambda tup: tup[1], reverse=True)]
        total = sum(pred_counter.values())
        table.append((expected, total, counts))
    # Sort by error sum
    table = sorted(table, reverse=True, key=lambda tup: tup[1])
    # Then, we expand lines
    output = []
    for word, total, errors in table:
        for index, (prediction, counter) in enumerate(errors):
            row = ["", ""]
            if index == 0:
                row = [word, total]
            row += [prediction, counter]
            output.append(row)
    return [["Expected", "Total Errors", "Predictions", "Predicted times"]] + output


def print_table(table):
    print(github_table.GithubFlavoredMarkdownTable(table).table)
