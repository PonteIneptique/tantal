import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tantal.models.pie import Pie
from tantal.data.train import GroundTruthDataset
from tantal.data.tokens.create import create_tokenizer, create_trainer
from tantal.data.tokens.train import parse_file
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

vocabulary = Vocabulary(
    tokenizer=tokenizer,
    tasks=[
        Task("lemma", categorical=False, unknown_ok=True),
        Task("pos", categorical=True, unknown_ok=False)
    ]
)
vocabulary.to_file("vocabulary.json")

train_dataset = GroundTruthDataset(TRAIN_FILE, vocabulary=vocabulary)
train_dataset.fit_vocab(max_lm_tokens=3000)
#train_dataset.downscale(.1)
train_loader = DataLoader(
    train_dataset,
    collate_fn=train_dataset.collate_fn,
    batch_size=128
)

dev_dataset = GroundTruthDataset(DEV_FILE, vocabulary=vocabulary)
#dev_dataset.downscale(.01)
dev_loader = DataLoader(
    dev_dataset,
    collate_fn=dev_dataset.collate_fn,
    batch_size=128
)
model = Pie(
    vocabulary,
    main_task="lemma",
    cemb_dim=100, cemb_layers=1, hidden_size=128, num_layers=2
)
trainer = pl.Trainer(gpus=1, max_epochs=50, gradient_clip_val=5)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
trainer.save_checkpoint("here3.model")
