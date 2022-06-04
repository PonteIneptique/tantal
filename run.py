import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar


from tantal.models.pie import Pie
from tantal.data.train import GroundTruthDataset
from tantal.data.tokens.create import create_tokenizer, create_trainer, train_for_bytes
from tantal.data.tokens.train import parse_file
from tantal.data.vocabulary import Vocabulary, Task
from tokenizers import Tokenizer

TOKENIZER_PATH = "fro.json"
TRAIN_FILE = "./exp_data/fro/train.tsv"
DEV_FILE = "./exp_data/fro/dev.tsv"
CHAR_LEVEL = True

if not os.path.exists(TOKENIZER_PATH):
    if CHAR_LEVEL:
        tokenizer = train_for_bytes(
            normalization="NFD",
            iterator_fn=parse_file,
            iterator_args=(TRAIN_FILE, ),
            special_tokens=["[UNK]", "[PAD]", "[EOS]", "[BOS]"]
        )
        tokenizer.save(TOKENIZER_PATH)
    else:
        tokenizer = create_tokenizer("bpe", "NFKD", "Whitespace,Digits")
        tokenizer_trainer = create_trainer("bpe", 500, special_tokens=["[UNK]", "[PAD]", "[EOS]", "[BOS]"])
        tokenizer.train_from_iterator(parse_file(TRAIN_FILE), trainer=tokenizer_trainer)
        tokenizer.save(TOKENIZER_PATH)
else:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

vocabulary = Vocabulary(
    tokenizer=tokenizer,
    tasks=[
        Task("lemma", categorical=False, unknown_ok=True),
        Task("POS", categorical=True, unknown_ok=True)
        #pos	Dis	Entity	Gend	Numb	Case	Deg	Mood_Tense_Voice	Person
    ]
)
train_dataset = GroundTruthDataset(TRAIN_FILE, vocabulary=vocabulary)
train_dataset.fit_vocab(max_lm_tokens=3000)
vocabulary.to_file("vocabulary.json")

#train_dataset.downscale(.01)
train_loader = DataLoader(
    train_dataset,
    collate_fn=train_dataset.collate_fn,
    batch_size=64
)

dev_dataset = GroundTruthDataset(DEV_FILE, vocabulary=vocabulary)
#dev_dataset.downscale(.1)
dev_loader = DataLoader(
    dev_dataset,
    collate_fn=dev_dataset.collate_fn,
    batch_size=64
)
model = Pie(
    vocabulary,
    main_task="lemma",
    cemb_dim=300, cemb_layers=2, hidden_size=256, num_layers=1
)
trainer = pl.Trainer(
    gpus=1,
    max_epochs=100,
    gradient_clip_val=5,
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="val_loss_main_task", patience=5, verbose=True),
        ModelCheckpoint(monitor="val_loss_main_task", save_top_k=2)
    ]
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
trainer.save_checkpoint("test.model")
