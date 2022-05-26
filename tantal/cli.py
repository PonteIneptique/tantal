import click
from tantal.readers.tab import TabReader


def _get_parser(mode: str):
    return TabReader


@click.group()
def tantal():
    """ Group of commands """


@tantal.group("tokenizer")
def tokenizer_cli():
    """ Tokenizer commands """


@tokenizer_cli.command("train")
@click.option("-m", "--parse-map", help="Map an extension to a specific parser")
def tokenizer_train():
    """ Train a tokenizer """
