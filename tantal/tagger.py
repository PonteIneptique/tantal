from tantal.models.pie import Pie
from tantal.data.vocabulary import Vocabulary
from tantal.data.batch import batchify_tokens
from typing import List


class PredictWrapper:
    def __init__(self, module):
        self.model: Pie = module
        self.model.eval()
        self.vocabulary: Vocabulary = module._vocabulary

    def predict_on_strings(self, sentences: List[List[str]], raw=False):
        data = []
        for sent in sentences:
            token, lengths = self.vocabulary.encode_input(sent)
            data.append({"token": token, "token__length": lengths})

        tokens, secondary_tasks = self.model(batchify_tokens(data, padding_value=self.vocabulary.token_pad_index))
        if raw:
            return tokens, secondary_tasks
        return tokens, {task: tensor.max(-1) for task, tensor in secondary_tasks.items()}
