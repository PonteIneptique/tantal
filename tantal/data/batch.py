from typing import List, Dict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def batchify(categorical_dict: List[Dict[str, List[int]]], padding_value: int) -> Dict[str, Tensor]:
    keys = list(categorical_dict[0].keys())
    return {
        key: pad_sequence(
            [
                torch.tensor(batch_element[key])
                for batch_element in categorical_dict
            ],
            batch_first=False,
            padding_value=padding_value
        )
        for key in keys
    }


def batchify_tokens(noncategorical_dict: List[Dict[str, List[List[int]]]], padding_value: int) -> Dict[str, Tensor]:
    keys = [key for key in noncategorical_dict[0].keys() if "length" not in key]
    return {
        **{
            key: pad_sequence(
                [
                    torch.tensor(subtoken_sequence)
                    for batch_element in noncategorical_dict
                    for subtoken_sequence in batch_element[key]
                ],
                batch_first=False,
                padding_value=padding_value
            )
            for key in keys
        }, **{
            f"{key}__length": torch.tensor([
                length
                for batch_element in noncategorical_dict
                for length in batch_element[f"{key}__length"]
            ])
            for key in keys
        }, **{
            f"{key}__sequence__length": torch.tensor(
                [len(batch_element[key]) for batch_element in noncategorical_dict]
            )
            for key in keys
        }
    }

