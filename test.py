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
TEST_FILE = "./exp_data/fro/test.tsv"
CHAR_LEVEL = True

vocabulary = Vocabulary.from_file("./saved_models/fro2/vocabulary.json")

test_dataset = GroundTruthDataset(TEST_FILE, vocabulary=vocabulary)
test_loader = DataLoader(
    test_dataset,
    collate_fn=test_dataset.collate_fn,
    batch_size=64
)
model = Pie.load_from_checkpoint(
    "lightning_logs/version_60/checkpoints/epoch=62-step=23688.ckpt",
    vocabulary=Vocabulary.from_file("saved_models/fro2/vocabulary.json"),
    main_task="lemma",
    cemb_dim=300, cemb_layers=2, hidden_size=150, num_layers=1,
    dropout=.3, cell="LSTM", char_cell="RNN", lr=0.0049,
    use_secondary_tasks_decision=True
)


trainer = pl.Trainer(
    gpus=1,
    max_epochs=100,
    gradient_clip_val=5,
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="acc_lemma_token_level", patience=5, verbose=True, mode="max"),
        ModelCheckpoint(monitor="acc_lemma_token_level", save_top_k=2, mode="max")
    ]
)
print(trainer.test(model, dataloaders=test_loader))
