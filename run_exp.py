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

TOKENIZER_PATH = "modernFR.json"
TRAIN_FILE = "./exp_data/fr/train/lemmat_full.tsv"
DEV_FILE = "./exp_data/fr/dev/lemmat_full.tsv"
TEST_FILE = None  # "./exp_data/fro/test.tsv"
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
        tokenizer = create_tokenizer("wordpiece", "NFD", "Whitespace,Digits")
        tokenizer_trainer = create_trainer("wordpiece", 400, special_tokens=["[UNK]", "[PAD]", "[EOS]", "[BOS]"])
        tokenizer.train_from_iterator(parse_file(TRAIN_FILE), trainer=tokenizer_trainer)
        tokenizer.save(TOKENIZER_PATH)
else:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)


TASKS = [
    Task("lemma", categorical=False, unknown_ok=True),
    # Task("POS", categorical=True, unknown_ok=True)
    #pos	Dis	Entity	Gend	Numb	Case	Deg	Mood_Tense_Voice	Person
]

vocabulary = Vocabulary(
    tokenizer=tokenizer,
    tasks=TASKS
)
train_dataset = GroundTruthDataset(
    TRAIN_FILE,
    vocabulary=vocabulary
)
train_dataset.fit_vocab(max_lm_tokens=20000)
vocabulary.to_file("vocabulary.json")

#train_dataset.downscale(.01)
train_loader = DataLoader(
    train_dataset,
    collate_fn=train_dataset.collate_fn,
    batch_size=128,
    shuffle=True
)

dev_dataset = GroundTruthDataset(DEV_FILE, vocabulary=vocabulary)
#dev_dataset.downscale(.1)
dev_loader = DataLoader(
    dev_dataset,
    collate_fn=dev_dataset.collate_fn,
    batch_size=128
)
if TEST_FILE:
    test_dataset = GroundTruthDataset(TEST_FILE, vocabulary=vocabulary)
    test_loader = DataLoader(test_dataset, collate_fn=dev_dataset.collate_fn, batch_size=64)


def create_trainer():
    return pl.Trainer(
        gpus=1,
        max_epochs=100,
        gradient_clip_val=5,
        callbacks=[
            TQDMProgressBar(),
            EarlyStopping(monitor="lemma_cer", patience=5, verbose=True, mode="min", min_delta=0.0005),
            ModelCheckpoint(monitor="lemma_cer", save_top_k=2, mode="min")
        ],
        move_metrics_to_cpu=True
    )


hyper_params = dict(
    cemb_dim=300, cemb_layers=2, hidden_size=150, num_layers=1,
    dropout=.25, cell="GRU", char_cell="RNN", lr=0.0049
)

for i in range(5):
    if len(TASKS) > 1:
        model = Pie(vocabulary, main_task="lemma", use_secondary_tasks_decision=True, mix_with_linear=True,
                    **hyper_params)
        trainer = create_trainer()
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
        trainer.save_checkpoint(f"FrWithUseMixed{i}.model")
        model.freeze()
        trainer.test(dataloaders=test_loader, model=model)

        del model, trainer

        model = Pie(vocabulary, main_task="lemma", use_secondary_tasks_decision=True, **hyper_params)
        trainer = create_trainer()
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
        trainer.save_checkpoint(f"FrWithUse{i}.model")
        model.freeze()
        trainer.test(dataloaders=test_loader, model=model)

        del model, trainer

    model = Pie(vocabulary, main_task="lemma", use_secondary_tasks_decision=False, **hyper_params)
    trainer = create_trainer()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    trainer.save_checkpoint(f"FrWithoutUse{i}-Secondary{len(TASKS)-1}.model")
    if TEST_FILE:
        trainer.test(dataloaders=test_loader, model=model)

    del model, trainer
