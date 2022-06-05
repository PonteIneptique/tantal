from tantal.models.pie import Pie
from tantal.data.vocabulary import Vocabulary
from tantal.tagger import PredictWrapper


model = Pie.load_from_checkpoint(
    "test.model",
    vocabulary=Vocabulary.from_file("saved_models/fro2/vocabulary.json"),
    main_task="lemma",
    cemb_dim=300, cemb_layers=2, hidden_size=256, num_layers=1
)



#tokens = [
#    "Ira enim in uindictam malorum sequi debet rationem animi non preire ut quasi ancilla iustitie"
#    " post tergum ueniat et non lasciua ante faciem prorumpat".split(),
#    ['Quid', 'faciat', 'uolt', 'scire', 'Lyris', 'quod', 'sobria', 'fellat']
#]
tokens = ["""Philippe de Thaün
Ad fait une raisun
Pur pruveires guarnir
De la lei maintenir. """, """
A sun uncle l'enveiet,
Quë amender la deiet
Si rien i ad mesdit
Ne en fait ne en escrit
A Unfrei de Thaün,
Le chapelein Yhun
E seneschal lu rei.""", """Icho vus di par mei."""]
tokens = tokens + ["Quel part que Bos irra irrunt", "Quant Bos ot od sei ces qu'il volt E espié e veü ot Liquels Petreïus esteit Ki tuz les altres mainteneit"]
tokens = [[sent.split() for sent in tokens][-1]]

print(tokens)
predict = PredictWrapper(model)
model.eval()
tokens, sec_tasks = predict.predict_on_strings(tokens)
print(predict.vocabulary.tokenizer.decode_batch(tokens.cpu().tolist()))
for task in sec_tasks:
    print(predict.vocabulary.decode_batch(sec_tasks[task].indices.cpu().tolist(), task=task))

# https://github.com/huggingface/tokenizers/issues/586