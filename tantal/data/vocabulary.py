from collections import namedtuple
from typing import List, Dict, Tuple, Union, Optional
import json


from tokenizers import Tokenizer, Encoding

Task = namedtuple("Task", ["name", "categorical", "unknown_ok"])

# ["PRE", "ADV"] -> [0, 1]
CategoricalSentenceEncoding = List[int]
# ["I", "am", "fine"] -> [[0], [1, 2], [3, 4, 5, 6]]
WordTokenVector = List[List[int]]
WordTokenSize = List[int]
WordTokenSentenceEncoding = Tuple[WordTokenVector, WordTokenSize]
AnySentenceEncodings = Union[CategoricalSentenceEncoding, WordTokenSentenceEncoding]


def get_word_groups(
        sentence: Encoding, bos: Optional[int] = None, eos: Optional[int] = None) -> WordTokenSentenceEncoding:
    """ Converts a sentence with pretokenized elements into a list of words

    """
    groups = [[bos] if bos else [] for i in range(max(sentence.word_ids) + 1)]
    for idx, word_id in enumerate(sentence.word_ids):
        groups[word_id].append(sentence.ids[idx])
    if eos:
        groups = [group + [eos] for group in groups]

    return groups, [len(group) for group in groups]


class Vocabulary:
    def __init__(
            self,
            tokenizer: Tokenizer,
            tasks: List[Task]
    ):
        self.tokenizer = tokenizer
        self.tasks: Dict[str, Task] = {
            task.name: task
            for task in tasks
        }
        if "token" in self.tasks:
            raise ValueError("Token cannot be a task, it's a reserved word for input tokens")

        self.tasks["lm_token"] = Task("lm_token", True, True)

        self.token_pad_index: int = self.tokenizer.token_to_id("[PAD]")
        self.token_bos_index: int = self.tokenizer.token_to_id("[BOS]")
        self.token_eos_index: int = self.tokenizer.token_to_id("[EOS]")

        self.tasks_vocab: Dict[str, Dict[str, int]] = {
            task.name: {"[PAD]": 0, "[UNK]": 1} if task.unknown_ok else {"[PAD]": 0}
            for task in self.tasks.values()
            if task.categorical
        }
        self.tasks_vocab_decoder: Dict[str, Tuple[str]] = dict()
        self._tokenizer_size = self.tokenizer.get_vocab_size()

    @property
    def tokenizer_size(self) -> int:
        return self._tokenizer_size

    def get_task_size(self, task: str) -> int:
        return len(self.tasks_vocab[task])

    def _build_reverse(self) -> None:
        self.tasks_vocab_decoder = {
            task: tuple(vocab.values())
            for task, vocab in self.tasks_vocab.items()
        }

    def encode_input(self, sequence: List[str]) -> WordTokenSentenceEncoding:
        return get_word_groups(
            sentence=self.tokenizer.encode(sequence, is_pretokenized=True),
            eos=self.token_eos_index,
            bos=self.token_bos_index
        )

    def encode(self, sequence: List[str], task: str) -> AnySentenceEncodings:
        if self.tasks[task].categorical:
            if self.tasks[task].unknown_ok:
                [self.tasks_vocab[task].get(element, "[UNK]") for element in sequence]
            else:
                return [self.tasks_vocab[task][element] for element in sequence]
        else:
            return get_word_groups(self.tokenizer.encode(sequence, is_pretokenized=True))

    def decode(self, sequence: List[int], task: str) -> List[str]:
        if self.tasks[task].categorical:
            return [self.tasks_vocab_decoder[task][codepoint] for codepoint in sequence]
        else:
            return self.tokenizer.decode(sequence, skip_special_tokens=True)

    def build_from_sentences(
        self,
        sentences: List[List[Dict[str, str]]]
    ) -> None:
        for sentence in sentences:
            for example in sentence:
                for task in self.tasks:
                    if example[task] not in self.tasks_vocab[task]:
                        self.tasks_vocab[task][example[task]] = len(self.tasks_vocab[task])
        self._build_reverse()
        return

    def dump(self) -> str:
        return json.dumps({
            "tokenizer": json.loads(self.tokenizer.to_str()),
            "tasks": [list(task) for task in self.tasks if task != "lm_token"],
            "vocabulary": self.tasks_vocab
        })

    def set_vocabulary(self, vocab: Dict[str, Dict[str, int]]):
        self.tasks_vocab = vocab
        self._build_reverse()

    def merge(self, vocabulary: "Vocabulary"):
        raise NotImplementedError("Will be implemented later to set-up some nice fiitting of models")

    @classmethod
    def from_string(cls, string) -> "Vocabulary":
        dictionary = json.loads(string)
        tokenizer = Tokenizer.from_str(json.dumps(dictionary["tokenizer"]))
        o = cls(
            tokenizer=tokenizer,
            tasks=[Task(*task) for task in dictionary["tasks"]]
        )
        o.set_vocabulary(dictionary["vocabulary"])
        return o