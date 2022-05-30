from typing import List, Optional
from tokenizers import Encoding


def get_word_groups(sentence: Encoding, bos: Optional[int] = None, eos: Optional[int] = None) -> List[List[int]]:
    """ Converts a sentence with pretokenized elements into a list of words

    """
    groups = [[bos] if bos else [] for i in range(max(sentence.word_ids) + 1)]
    for idx, word_id in enumerate(sentence.word_ids):
        groups[word_id].append(sentence.ids[idx])
    if eos:
        return [group + [eos] for group in groups]
    return groups
