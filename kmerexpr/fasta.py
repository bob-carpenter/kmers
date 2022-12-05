"""
Module for reading in FASTA sequence files.

Structured similarly to BioPython's BSD-3 parser
https://github.com/biopython/biopython/blob/76796c970a68fa09da94dc95827b217a872004e2/Bio/SeqIO/FastaIO.py
"""

from dataclasses import dataclass
from typing import Generator
from io import TextIOBase


@dataclass
class Sequence:
    header: str
    sequence: str


def read_fasta(handle: TextIOBase) -> Generator[Sequence, None, None]:
    """
    Basic lazy FASTA format reader.

    This assumes a single description line per sequence. Additional
    comments are not supported.

    Sequences can be split over multiple lines, and whitespace lines
    are ignored.
    """
    head = ""
    for line in handle:
        if line.startswith(">"):
            head = line[1:].rstrip()
            break
    if not head:  # empty file
        return

    seq = ""
    for line in handle:
        if line.startswith(">"):  # start of next gene
            yield Sequence(head, seq)
            seq = ""
            head = line[1:].rstrip()
            continue
        seq += line.strip()

    yield Sequence(head, seq)
