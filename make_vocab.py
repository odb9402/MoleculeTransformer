import torch
import torch.nn
import torchtext
import random
import re
import multiprocessing as mp
from molecule_transformer_trainer import MoleculeTransformerTrainer
from torchtext.data.utils import get_tokenizer
from collections import Counter
import subprocess as sp

import argparse

parser = argparse.ArgumentParser(description="Molecule transformer training")
parser.add_argument("-i", "--input", help="Input training raw SMILES file.")
parser.add_argument("-o", "--output", help="Vocabulary file")
trainer = MoleculeTransformerTrainer()

dataset = torchtext.data.TabularDataset(args.input,
                                        format='csv',
                                        fields=[('input', trainer.smile_mol_tokenizer),                                                                 ('output', trainer.smile_mol_masked_tokenizer)])
trainer.smile_mol_tokenizer.build_vocab(dataset)
trainer.save_vocab(args.output)