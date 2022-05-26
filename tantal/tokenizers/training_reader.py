import re
import glob
from typing import List


def stringify(sentence: List[str]) -> str:
    return re.sub(r"([\(\)'\"]) | ([\.\:;,\"])", "\g<2>\g<1>", " ".join(sentence))


def parse(paths=(
    "../../exp_data/OF3C/tsv/*.tsv",
    "../../exp_data/word_segmentation_data/fro/src/bfm/txt/*.txt",
    "../../exp_data/word_segmentation_data/fro/src/nca/txt/*.txt")):
    for path in paths:
        for file in glob.glob(path):
            out_form, out_lemma = [[]], [[]]
            with open(file) as f:
                if file.endswith(".tsv"):
                    for idx, line in enumerate(f):
                        if idx == 0:
                            continue
                        if len(line.strip().split()) > 2:
                            form, lemma, *_ = line.strip().split()
                            out_form[-1].append(form)
                            out_lemma[-1].append(lemma)
                        elif out_form[-1]:
                            out_form.append([])
                            out_lemma.append([])
                    yield from [stringify(sentence) for sentence in out_form]
                    yield from [stringify(sentence) for sentence in out_lemma]
                elif file.endswith(".txt"):
                    yield from re.split("(\.!?) ", f.read())