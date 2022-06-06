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

tokenizer = train_for_bytes(
    iterator_fn=parse_file,
    iterator_args=(TRAIN_FILE, ),
    special_tokens=["[UNK]", "[PAD]", "[EOS]", "[BOS]"]
)
tokenizer.save(TOKENIZER_PATH)
