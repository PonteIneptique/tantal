from tantal.models.pie import Pie
from tantal.data.vocabulary import Vocabulary
from tantal.tagger import PredictWrapper


model = Pie.load_from_checkpoint(
    "heavier.model",
    vocabulary=Vocabulary.from_file("vocabulary.json"),
    main_task="lemma",
    cemb_dim=200, cemb_layers=2, hidden_size=256, num_layers=1
)


tokens = [
    "Ira enim in uindictam malorum sequi debet rationem animi non preire ut quasi ancilla iustitie post tergum ueniat et non lasciua ante faciem prorumpat".split()
]

print(tokens)
predict = PredictWrapper(model)
model.eval()
tokens, sec_tasks = predict.predict_on_strings(tokens)
print(predict.vocabulary.tokenizer.decode_batch(tokens.cpu().tolist()))
print(predict.vocabulary.decode_batch(sec_tasks["pos"].indices.cpu().tolist(), task="pos"))

# https://github.com/huggingface/tokenizers/issues/586