from dataclasses import dataclass
from typing import Union, Dict
from torch import Tensor

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score


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
