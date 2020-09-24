import torch
import torch.nn
import torchtext
import random
import re
from torchtext.data.utils import get_tokenizer
import argparse

parser = argparse.ArgumentParser(description="Molecule transformer training")
parser.add_argument("-i", "--input", default="CID-SMILES.txt", help="Input training raw SMILES file.")
parser.add_argument("-o", "--output", default="CID-SMILES_train.txt", help="Comma separated training SMILES file.")
args = parser.parse_args()

input_file = args.input #'./CID-SMILES.txt'
masked_output_file = args.output #'./CID-SMILES_train_valid.txt'

def valid_SMILES(mol_str):
    template = re.compile('(^([^J][0-9BCOHNSOPrIFla@+\-\[\]\(\)\\\/%=#$]{6,})$)', re.I)
    match = template.match(mol_str)
    return bool(match)

smile_mol_tokenizer = torchtext.data.Field(init_token='<BEGIN>',
                                          pad_token='<PAD>',
                                          fix_length=True,
                                          tokenize=list,
                                          eos_token='<END>')

smile_data = torchtext.data.TabularDataset(path=input_file,
                                          format='tsv',
                                          fields=[('smile_mol', smile_mol_tokenizer)])

train_data, test_data = smile_data.split(split_ratio=0.7)
smile_mol_tokenizer.build_vocab(smile_data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

possible_tokens = smile_mol_tokenizer.vocab.itos[4:]
input_data = open(input_file, 'r')
masked_output_file = open(masked_output_file, 'w')

for mol in input_data:
    mol = mol.rstrip()
    if not valid_SMILES(mol):
        continue
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
    masked_output_file.write("$"+masked_mol+"$"+","+"$" + output_mol +"$"+ "\n") ## "$" as the <BEGIN> and <END> token

masked_output_file.close()
