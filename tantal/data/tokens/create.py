import tokenizers
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, ByteLevelBPETokenizer
from typing import Dict, Type, Optional, List, Union
from unicodedata import normalize


AvailableModels: Dict[str, Type[models.Model]] = {
    "unigram": models.Unigram,
    "bpe": models.BPE,
    "wordpiece": models.WordPiece
}

AvailableTrainer: Dict[str, Type[trainers.Trainer]] = {
    "unigram": trainers.UnigramTrainer,
    "bpe": trainers.BpeTrainer,
    "wordpiece": trainers.WordPieceTrainer
}


class UnknownTokenizerModel(ValueError):
    """ Error raised when a model does not exist for tokenizers"""


class UnknownNormalizer(ValueError):
    """ Error raised when a normalizer does not exist for tokenizers"""


class UnknownPreTokenizer(ValueError):
    """ Error raised when a pretokenizer does not exist for tokenizers"""


def get_normalizer(normalization: str) -> normalizers.Normalizer:
    def normalizer_exists(norm: str):
        if hasattr(normalizers, norm):
            return True
        raise UnknownNormalizer(norm)

    if "," in normalization:
        return normalizers.Sequence([
            getattr(normalizers, norm)()
            for norm in normalization.split(",")
            if normalizer_exists(norm)
        ])
    elif normalizer_exists(normalization):
        return getattr(normalizers, normalization)()


def get_pretokenizer(tokenization: str) -> tokenizers.Tokenizer:
    def pretokenizer_exists(t: str):
        if hasattr(pre_tokenizers, t):
            return True
        raise UnknownPreTokenizer(t)

    if "," in tokenization:
        return pre_tokenizers.Sequence([
            getattr(pre_tokenizers, norm)()
            for norm in tokenization.split(",")
            if pretokenizer_exists(norm)
        ])
    elif pretokenizer_exists(tokenization):
        return getattr(pre_tokenizers, tokenization)()


def create_tokenizer(
        model: Union[str, Tokenizer],
        normalization: Optional[str] = None,
        pretokenizer: Optional[str] = None) -> Tokenizer:
    """ CLI helper for creating a Tokenizer

    :param model: Name of the tokenizing model
    :param normalization: String where steps of normalization are separated by commas, eg. "NFKD,Lowercase"
    :param pretokenizer: String where steps of pretokenization are separated by commas, eg. "Whitespace,Digits"

    """
    if not isinstance(model, (ByteLevelBPETokenizer, Tokenizer)):
        if model.lower() not in AvailableModels:
            raise UnknownTokenizerModel(model)
        tokenizer = Tokenizer(AvailableModels[model]())
    else:
        tokenizer = model

    if normalization:
        tokenizer.normalizer = get_normalizer(normalization)

    if pretokenizer:
        tokenizer.pre_tokenizer = get_pretokenizer(pretokenizer)
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    if isinstance(model, str):

        if model.lower() == "bpe":
            tokenizer.decoder = decoders.BPEDecoder()
        elif model.lower() == "wordpiece":
            tokenizer.decoder = decoders.WordPiece()
        else:
            tokenizer.decoder = decoders.ByteLevel()
    else:
        tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def train_for_bytes(
    iterator_fn,
    iterator_args,
    special_tokens: List[str] = None,
    canonical: bool = True
) -> Tokenizer:
    #tokenizer = ByteLevelBPETokenizer(vocab=[], add_prefix_space=False, unicode_normalizer="nfkd")
    chars = set()
    normalize_mode = "NFD"
    if canonical:
        normalize_mode = "NFKD"
    for sentence in iterator_fn(*iterator_args):
        chars = chars.union(set(normalize(normalize_mode, sentence+sentence.upper())))

    chars = sorted(list("".join(list(chars)).strip()))

    tokenizer = ByteLevelBPETokenizer(
        vocab={
            char: idx
            for idx, char in enumerate(special_tokens+chars)
        },
        merges=[],
        add_prefix_space=False,
        unicode_normalizer=normalize_mode.lower()
    )
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.model.unk_token = "[UNK]"
    return tokenizer


def create_trainer(model: str, vocabulary: int, special_tokens: List[str]) -> trainers.Trainer:
    """ Create a trainer for a specific model

    """
    if model.lower() not in AvailableTrainer:
        raise UnknownTokenizerModel(model)
    return AvailableTrainer[model](vocab_size=vocabulary, special_tokens=special_tokens, unk_token="[UNK]")


if __name__ == "__main__":
    from .train import parse
    tok = create_tokenizer("wordpiece", "NFKC,Lowercase", "Whitespace,Digits")
    tok.train_from_iterator(parse(), trainer=create_trainer("wordpiece", 20000, [[],"<BOS>", "<EOS>", "<PAD>"]))
    print(tok)
