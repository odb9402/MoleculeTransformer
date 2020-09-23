from molecule_transformer_trainer import MoleculeTransformerTrainer
from model import MoleculeTransformer
import argparse
import sys

parser = argparse.ArgumentParser(description="Molecule transformer training")
parser.add_argument("-i", "--input", help="Input training SMILES file.")
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("--lossWeight", choices=['none','log','sqrt','raw'], default='raw',
                    help="The type of class weights for the cross entropy loss.")
args = parser.parse_args()

if args.input == None:
    parser.print_help()
    sys.exit()

trainer = MoleculeTransformerTrainer(args.input, class_weight=args.lossWeight)

trainer.build_model()
trainer.print_params()
print("Training processes . . . .")

for i in range(args.epochs):
    trainer.train(log='stdout')
    trainer.save_model("model{}".format(i))
    acc = trainer.evaluate_acc()
    print("ACCURACY of epoch {} = {}".format(acc))
trainer.export_training_figure()
