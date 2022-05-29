from typing import List
from tokenizers import Encoding


def get_word_groups(sentence: Encoding) -> List[List[int]]:
    """ Converts a sentence with pretokenized elements into a list of words

    """
    groups = [[] for i in range(max(sentence.word_ids) + 1)]
    for idx, word_id in enumerate(sentence.word_ids):
        groups[word_id].append(sentence.ids[idx])
    return groups
