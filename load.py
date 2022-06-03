from tantal.models.pie import Pie
from tantal.data.vocabulary import Vocabulary
from tantal.data.batch import batchify_tokens
from typing import List

model = Pie.load_from_checkpoint(
    "here3.model",
    vocabulary=Vocabulary.from_file("vocabulary.json"),
    main_task="lemma",
    cemb_dim=100, cemb_layers=1, hidden_size=128, num_layers=2
)
# model.load_from_checkpoint() Need implementation ?

#Vocabulary.from_file("vocabulary.json")
#data = model._vocabulary.encode_input(["Ego", "sum", "pulcher", "."])
#print(data)


class PredictWrapper:
    def __init__(self, module):
        self.model: Pie = module
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

# model._vocabulary.tokenizer.model.unk_id = model._vocabulary.tokenizer.token_to_id("[UNK]")
tokens = [
    "Quid faciat volt scire Maria quod sobria fellat".split()
]
print(tokens)
predict = PredictWrapper(model)
tokens, pos = predict.predict_on_strings(tokens)
print(predict.vocabulary.tokenizer.decode_batch(tokens.cpu().tolist()))
print(predict.vocabulary.decode_batch(pos["pos"].indices.cpu().tolist(), task="pos"))
