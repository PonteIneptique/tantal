from tantal.models.pie import Pie
from tantal.data.vocabulary import Vocabulary
from tantal.tagger import PredictWrapper


model = Pie.load_from_checkpoint(
    "./test.model",
    vocabulary=Vocabulary.from_file("vocabulary.json"),
    main_task="lemma",
    cemb_dim=200, cemb_layers=2, hidden_size=256, num_layers=1
)


tokens = [
    "Ira enim in uindictam malorum sequi debet rationem animi non preire ut quasi ancilla iustitie"
    " post tergum ueniat et non lasciua ante faciem prorumpat".split(),
    ['Quid', 'faciat', 'uolt', 'scire', 'Lyris', 'quod', 'sobria', 'fellat']
]

print(tokens)
predict = PredictWrapper(model)
model.eval()
tokens, sec_tasks = predict.predict_on_strings(tokens)
print(predict.vocabulary.tokenizer.decode_batch(tokens.cpu().tolist()))
for task in sec_tasks:
    print(predict.vocabulary.decode_batch(sec_tasks[task].indices.cpu().tolist(), task=task))

# https://github.com/huggingface/tokenizers/issues/586