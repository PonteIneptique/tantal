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

TOKENIZER_PATH = "fro3.json"
TRAIN_FILE = "./exp_data/fro/train.tsv"
DEV_FILE = "./exp_data/fro/dev.tsv"
TEST_FILE = "./exp_data/fro/test.tsv"
CHAR_LEVEL = True

if not os.path.exists(TOKENIZER_PATH):
    if CHAR_LEVEL:
        tokenizer = train_for_bytes(
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
train_dataset.fit_vocab(max_lm_tokens=10000)
vocabulary.to_file("vocabulary.json")

#train_dataset.downscale(.01)
train_loader = DataLoader(
    train_dataset,
    collate_fn=train_dataset.collate_fn,
    batch_size=64,
    shuffle=True
)

dev_dataset = GroundTruthDataset(DEV_FILE, vocabulary=vocabulary)
#dev_dataset.downscale(.1)
dev_loader = DataLoader(
    dev_dataset,
    collate_fn=dev_dataset.collate_fn,
    batch_size=64
)
test_dataset = GroundTruthDataset(TEST_FILE, vocabulary=vocabulary)
test_loader = DataLoader(test_dataset, collate_fn=dev_dataset.collate_fn, batch_size=64)


def create_trainer():
    return pl.Trainer(
        gpus=1,
        max_epochs=100,
        gradient_clip_val=5,
        callbacks=[
            TQDMProgressBar(),
            EarlyStopping(monitor="lemma_pre", patience=5, verbose=True, mode="max"),
            ModelCheckpoint(monitor="lemma_pre", save_top_k=2, mode="max")
        ],
        amp_level="01"
    )

hyper_params = dict(
        cemb_dim=300, cemb_layers=2, hidden_size=150, num_layers=1,
        dropout=.3, cell="LSTM", char_cell="RNN", lr=0.0049
)

for i in range(5):
    model = Pie(vocabulary, main_task="lemma", use_secondary_tasks_decision=True, **hyper_params)
    trainer = create_trainer()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    trainer.save_checkpoint(f"FroWithUse{i}.model")
    model.freeze()
    trainer.test(dataloaders=test_loader, model=model)

    del model, trainer

    model = Pie(vocabulary, main_task="lemma", use_secondary_tasks_decision=False, **hyper_params)
    trainer = create_trainer()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    trainer.save_checkpoint(f"FroWithoutUse{i}.model")
    trainer.test(dataloaders=test_loader, model=model)

    del model, trainer
