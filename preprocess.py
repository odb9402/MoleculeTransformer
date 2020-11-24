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
parser.add_argument("-i", "--input", default="CID-SMILES.txt", help="Input training raw SMILES file.")
parser.add_argument("-o", "--output", default="CID-SMILES_train_tok.txt", help="Comma separated training SMILES file.")
parser.add_argument("-v", "--vocab", default="preproc_vocab.voc", help="Saved torch tensor file for the vocab.")
parser.add_argument("--experimental", action='store_true', help="Using experimental tokenization: See the molecule_transformer_trainer.py")
args = parser.parse_args()

input_file = args.input #'./CID-SMILES.txt'
masked_output_file = args.output #'./CID-SMILES_train_valid.txt'

def valid_SMILES(mol_str):
    template = re.compile('(^([^J][0-9BCOHNSOPrIFla@+\-\[\]\(\)\\\/%=#$]{6,})$)', re.I)
    match = template.match(mol_str)
    return bool(match

"""
result = sp.check_output(["sh", "count.sh", args.input])
counter = Counter()
for line in result.decode('utf-8').split('\n'):
    if line.rstrip('\n') != '':
        key, count = line.split()
        counter[key] = int(count)
vocab = torchtext.vocab.Vocab(counter, specials=['<unk>', '<PAD>', '<REP>',' ']) ## ' ' is the mask
torch.save(vocab, args.vocab) ## Save vocab file
possible_tokens = vocab.itos[2:]
"""

if args.experimental:
    tokenize_func = MoleculeTransformerTrainer.tokenize_train_new
else:
    tokenize_func = list
    
smile_mol_tokenizer = torchtext.data.Field(init_token='<REP>',
                                          pad_token='<PAD>',
                                          tokenize=tokenize_func)#list)
smile_data = torchtext.data.TabularDataset(path=input_file,
                                          format='tsv',
                                          fields=[('smile_mol', smile_mol_tokenizer)])
smile_mol_tokenizer.build_vocab(smile_data)
torch.save(smile_mol_tokenizer.vocab, args.vocab) ## Save vocab file
possible_tokens = smile_mol_tokenizer.vocab.itos[3:] ## Watch out! 
print(possible_tokens)

input_data = open(input_file, 'r')
masked_output_file = open(masked_output_file, 'w')

io_step = 0
io_string = ''

for mol in input_data:
    if not valid_SMILES(mol.rstrip()):
        continue
    if args.experimental:
        mol = MoleculeTransformerTrainer.tokenize_train_new(mol)
    else:
        mol = mol.rstrip()
        mol = list(mol)
    
    masked_mol = ""
    output_mol = ""
    for i in range(len(mol)):
        if random.random() < 0.15:
            if random.random() < 0.8:
                masked_mol += " " ### " " means the [MASK].
            else:
                if random.random() < 0.5:
                    masked_mol += random.choice(possible_tokens)
                else:
                    masked_mol += mol[i]
            output_mol += mol[i]
        else:
            masked_mol += mol[i] ### Ignore sequence &. The index of & will be ignored when calculate the loss.
            output_mol += "&"
    io_step += 1
    io_string += "$" + masked_mol + "$" + "," + "$" + output_mol + "$" + "\n"
    
    if io_step % 100000 == 0 and io_step != 0:
        masked_output_file.write(io_string)
        masked_output_file.flush()
        io_string = ''
        #masked_output_file.write("$"+masked_mol+"$"+","+"$" + output_mol +"$"+ "\n") ## "$" as the <BEGIN> and <END> token

if io_string != '':
    masked_output_file.write(io_string)
    masked_output_file.flush()
masked_output_file.close()
