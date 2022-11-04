from typing import List, Tuple, Dict, Optional

import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer, Encoding

from tantal.data.vocabulary import Task, Vocabulary, get_word_groups
from tantal.data.batch import batchify, batchify_tokens
from tantal.noise import UppercaseNoise, BaseNoise


class GroundTruthDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        vocabulary: Vocabulary,
        noise: Optional[List[BaseNoise]] = None
    ):
        self.annotation_file: str = annotations_file
        self.vocabulary: Vocabulary = vocabulary
        self.tasks: Dict[str, Task] = vocabulary.tasks
        self.tokenizer: Tokenizer = vocabulary.tokenizer

        self.annotations: List[Dict[str, List[str]]] = list(self._read())
        self.pad_index = self.tokenizer.token_to_id("[PAD]")
        self.bos_index = self.tokenizer.token_to_id("[BOS]")
        self.eos_index = self.tokenizer.token_to_id("[EOS]")

        self.noise_creators: List[BaseNoise] = noise or []

    def downscale(self, ratio: float, shuffle: bool = False) -> int:
        end = int(len(self.annotations) * ratio)
        if shuffle:
            random.shuffle(self.annotations)
        self.annotations = self.annotations[:end]
        return end

    def _read(self):
        if self.annotation_file.endswith(".tsv"):
            return self._read_tsv(self.annotation_file, list(self.tasks.keys()))
        raise ValueError("Unsupported data format")

    def fit_vocab(self, max_lm_tokens: Optional[int] = None):
        self.vocabulary.build_from_sentences(self.annotations, max_lm_tokens=max_lm_tokens)

    @staticmethod
    def _read_tsv(filepath, tasks):
        with open(filepath) as f:
            sentence = []
            headers = []
            for line_idx, line in enumerate(f):
                line = line.strip().split("\t")
                if line_idx == 0:
                    headers = line
                elif not [col for col in line if col]:
                    if sentence:
                        yield {
                            task: [token[task] for token in sentence]
                            for task in sentence[0]
                        }
                        sentence = []
                else:
                    line = dict(zip(headers, line))
                    try:
                        sentence.append({
                            "token": line["token"],
                            "lm_token": line["token"],
                            **{
                                task: line[task]
                                for task in tasks
                                if task != "lm_token"
                            }
                        })
                    except KeyError:
                        raise  # Custom one day ?

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sentence: Dict[str, List[str]] = self.annotations[idx]

        tokens = sentence["token"]
        for noise_creator in self.noise_creators:
            if noise_creator.corresponds(tokens) and noise_creator.random():
                tokens = noise_creator.apply(tokens)

        tokens, token_length = self.vocabulary.encode_input(sentence["token"])

        categoricals = {
            task: self.vocabulary.encode(sentence[task], task=task)
            for task in self.vocabulary.tasks
            if task != "token" and self.tasks[task].categorical
        }

        non_categoricals = {
            key: value
            for task in self.vocabulary.tasks
            if task != "token" and not self.tasks[task].categorical
            for key, value in dict(zip(
                [task, f"{task}__length"],
                self.vocabulary.encode(sentence[task], task=task)
            )).items()
        }

        return {
            "token": tokens,
            "token__length": token_length
               }, {
            "categoricals": categoricals,
            "non_categoricals": non_categoricals
        }

    def collate_fn(self, batch):
        """
        DataLoaderBatch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        tokens, gt = zip(*batch)
        return batchify_tokens(tokens, self.vocabulary.token_pad_index), {
            "categoricals": batchify(
                [ex["categoricals"] for ex in gt],
                padding_value=self.vocabulary.categorical_pad_token_index
            ),
            "non_categoricals": batchify_tokens(
                [ex["non_categoricals"] for ex in gt],
                padding_value=self.vocabulary.token_pad_index
            )
        }
