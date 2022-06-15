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

TOKENIZER_PATH = "./saved_models/fro2/fro.json"
TRAIN_FILE = "./exp_data/fro/train.tsv"
DEV_FILE = "./exp_data/fro/dev.tsv"
TEST_FILE = "./exp_data/fro/digluville.tsv"
CHAR_LEVEL = True

#vocabulary = Vocabulary.from_file("./saved_models/fro2/vocabulary.json")

models = []

model = Pie.load_from_checkpoint(
    "/home/thibault/dev/tantal/lightning_logs/version_210/checkpoints/epoch=44-step=16920.ckpt",
    hparams_file="/home/thibault/dev/tantal/lightning_logs/version_210/hparams.yaml",
    vocabulary=Vocabulary.from_file("/home/thibault/dev/tantal/lightning_logs/version_211/vocabulary.json")
)
model.freeze()
models.append(model)


trainer = pl.Trainer(
    gpus=1,
    max_epochs=100,
    gradient_clip_val=5,
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="acc_lemma", patience=5, verbose=True, mode="max"),
        ModelCheckpoint(monitor="acc_lemma", save_top_k=2, mode="max")
    ]
)

for model in models:
    print(model.hparams)
    for test_file in ["./exp_data/fro/digluville.tsv"]:#, "./exp_data/fro/nca.tsv", "./exp_data/fro/calais.tsv"]:
        test_dataset = GroundTruthDataset(test_file, vocabulary=model._vocabulary)
        test_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate_fn, batch_size=64)
        print(trainer.test(model, dataloaders=test_loader))
