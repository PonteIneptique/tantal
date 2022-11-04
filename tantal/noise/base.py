import random
from typing import List, Dict, Tuple


Data = Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]


class BaseNoise:
    def __init__(self, ratio: float = .05):
        super(BaseNoise, self).__init__()
        self.ratio = ratio

    def random(self) -> bool:
        return self.ratio < random.random()

    def corresponds(self, tokens: List[str]) -> bool:
        return True

    def apply(self, tokens: List[str], categorical, non_categorical) -> List[str]:
        raise NotImplementedError('Main class cannot be called')


class UppercaseNoise(BaseNoise):
    def apply(self, tokens: List[str], categorical, non_categorical) -> List[str]:
        return [token.upper() for token in tokens], categorical, non_categorical


class RandomCut(BaseNoise):
    def __init__(self, ratio: float = .01, min_tokens: int = 3):
        """ This noiser cuts a sentence. This helps creating models that agrees with shorten context, specifically
    situation like HTR where some data could be lost.

    :param ratio: The chance of the action to be applied
    :param min_tokens: The minimum amount of tokens in a sentence

    """
        super(RandomCut, self).__init__(ratio=ratio)
        self.min_tokens = min_tokens

    def corresponds(self, tokens: List[str]) -> bool:
        return len(tokens) > self.min_tokens

    def apply(self, tokens: List[str], categorical, non_categorical) -> List[str]:
        length = random.randint(self.min_tokens, len(tokens))
        start = random.randint(0, len(tokens)-length)
        return tokens[start:start+length]
