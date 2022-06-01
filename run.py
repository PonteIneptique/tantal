import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tantal.modules.pie.main import Pie
from tantal.data.train import GroundTruthDataset
from tantal.data.vocabulary import Vocabulary, Task
from tokenizers import Tokenizer

TOKENIZER_PATH = "./example-tokenizer.json"
TRAIN_FILE = "../LASLA/mood-tense-voice-pft-clitics/train.tsv"
DEV_FILE = "../LASLA/mood-tense-voice-pft-clitics/dev.tsv"

if not os.path.exists(TOKENIZER_PATH):
    tokenizer = create_tokenizer("unigram", "NFKD,Lowercase", "Whitespace,Digits")
    tokenizer_trainer = create_trainer("unigram", 1000, special_tokens=["[PAD]", "[EOS]", "[BOS]"])
    tokenizer.train_from_iterator(parse_file(TRAIN_FILE), trainer=tokenizer_trainer)
    tokenizer.save(TOKENIZER_PATH)
else:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

voc = Vocabulary(
    tokenizer=tokenizer,
    tasks=[
        Task("lemma", categorical=False, unknown_ok=True),
        Task("pos", categorical=True, unknown_ok=False)
    ]
)

train_dataset = GroundTruthDataset(TRAIN_FILE, vocabulary=voc)
train_dataset.fit_vocab()

print(train_dataset[0])

raise
train_loader = DataLoader(
    train_dataset,
    collate_fn=train_dataset.collate_fn,
    batch_size=4
)
dev_dataset = GroundTruthDataset(DEV_FILE, main_task="lemma", tokenizer=tokenizer)
dev_loader = DataLoader(
    dev_dataset,
    collate_fn=dev_dataset.collate_fn,
    batch_size=4
)
#model = Pie(tokenizer, cemb_dim=50, cemb_layers=1, hidden_size=128, num_layers=2)
trainer = pl.Trainer(gpus=1)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
