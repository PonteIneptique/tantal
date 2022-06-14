from dataclasses import dataclass
from typing import Union, Dict
from torch import Tensor


@dataclass
class ScoreWatcher:
    """Class for keeping track of an item in inventory."""
    score: Union[Tensor, float, int]
    mode: str = "min"
    steps: int = 0
    patience: int = 5
    delta: float = 0.05
    factor: float = 0.6
    min_weight: float = .2
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
                    weight = weight
                    print(f"Minimal weight reached")
                else:
                    print(f"Weights have been updated for task `{task}` (Score: `{self.score}`, Weight: `{weight}` )")
                self.steps = 0
        return self.steps, weight

    def repr(self, task_name: str, weight: float):
        return f'<ScoreWatcher task="{task_name}" weight="{weight:.2f}" ' \
               f'best="{self.score:.2f}" mode="{self.mode}" />'
