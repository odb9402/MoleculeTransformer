from molecule_transformer_trainer import MoleculeTransformerTrainer
from model import MoleculeTransformer
import argparse
import sys
import glob

parser = argparse.ArgumentParser(description="Molecule transformer training")
parser.add_argument("-i", "--input", help="Input training SMILES file.")
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-v", "--vocab", help="Vocabulary file")
parser.add_argument("--lossWeight", choices=['none','log','sqrt','raw'], default='none',
                    help="The type of class weights for the cross entropy loss.")
args = parser.parse_args()

if args.input == None:
    parser.print_help()
    sys.exit()

trainer = MoleculeTransformerTrainer(args.input,
                                     class_weight=args.lossWeight,
                                     vocab_file=args.vocab,
                                     n_tokens=415)

print("Dataset split. . . .")
directory = MoleculeTransformerTrainer.split_file(args.input, line_num=1000000)
datasets = glob.glob(directory + "/*")
print("Splited datasets:" + str(datasets))

if args.vocab != None:
    trainer.load_vocab()
    print(trainer.smile_mol_tokenizer.vocab.stoi)
    print(len(trainer.smile_mol_tokenizer.vocab.stoi))

print("Training processes . . . .")

build=False
for i in range(args.epochs):
    for data in datasets:
        print("Load the dataset {}".format(data))
        trainer.gen_dataloader(data)
        print("Training for splited file {}".format(data))
        if not build:
            trainer.build_model()
            trainer.model_summary()
            build = True
        trainer.train(log='stdout')
    trainer.save_model("model{}".format(i+1))
    acc = trainer.evaluate_acc()
    print("ACCURACY of epoch {} = {}".format(i+1, acc))
    trainer.export_training_figure(i)
