
# Molecule transformer using BERT based model

The BERT-based embedding model SMILES molecule representation from the paper ["Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction"](http://proceedings.mlr.press/v106/shin19a.html) written by Shin et al 2019. These sources are pytorch-implemented codes.

## Usage
```python
awk '{OFS="\t"; FS="\t"; print $2}' CID-SMILES > CID-SMILES.txt
python preprocess.py -i CID-SMILES.txt -o CID-SMILES_train.txt
python main.py -i CID-SMILES_train.txt -e 5 --lossWeight none
```
```
usage: main.py [-h] [-i INPUT] [-e EPOCHS] [--lossWeight {none,log,sqrt,raw}]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input training SMILES file.
  -e EPOCHS, --epochs EPOCHS
  --lossWeight {none,log,sqrt,raw}
                        The type of class weights for the cross-entropy loss.
```

### Final accuracy
- Epochs 1, without loss weights: 0.9471
-
