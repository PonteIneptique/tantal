from tantal.models.pie import Pie
from tantal.data.vocabulary import Vocabulary


model = Pie.load_from_checkpoint(
    "here2.model",
    vocabulary=Vocabulary.from_file("vocabulary.json"),
    main_task="lemma",
    cemb_dim=100, cemb_layers=1, hidden_size=128, num_layers=2
)
# model.load_from_checkpoint() Need implementation ?


data = model._vocabulary.encode_input(["Ego", "sum", "pulcher", "."])
print(data)

