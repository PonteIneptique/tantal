from tantal.data.vocabulary import Vocabulary

voc = Vocabulary.from_file("vocabulary.json")
data = voc.encode_input(["Ego", "sum", "pulcher", "."])
print(voc.get_task_size("pos"))
