from model import *
import re
import time
import torch
import torch.nn
import torchtext
from torchtext.data import Iterator
from torchtext.data.utils import get_tokenizer
import random
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

class MoleculeTransformerTrainer():
    """
    Attrs:

    Methods:

    """
    def __init__(self, train_file, class_weight='none'):
        self.mol_emsize = 128 # Embedded molecule sizes
        self.n_layers = 8 # Number of attentions and feed-forwards
        self.n_head = 8 # Attention heads
        self.n_hid = 512 # feed forward dim
        self.lr = 0.0001
        self.epochs = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weight=class_weight

        print("Generate tokenizers. . . .")
        self.gen_tokenizers()
        print("Generate dataloaders. . . .")
        dataload_time = time.time()
        self.gen_dataloader(train_file)
        self.n_tokens = 73#len(self.smile_mol_tokenizer.vocab.stoi)
        print("Data loading finished : {:5.2f}sec".format(time.time()-dataload_time))

        token_idx = self.smile_mol_tokenizer.vocab.stoi
        self.ignored_token = token_idx['<unk>']
        self.mask_token = token_idx[' ']
        self.untargeted_tokens = ['<unk>', '<PAD>', '<REP>', '$', ' ']

        self.loss_history = []
        self.acc_history = []

    def gen_tokenizers(self):
        self.smile_mol_tokenizer = torchtext.data.Field(init_token='<REP>', ### $ is the [BEGIN]
                                                  pad_token='<PAD>',
                                                  tokenize=list,
                                                  fix_length=100,
                                                  batch_first=True)

        self.smile_mol_masked_tokenizer = torchtext.data.Field(fix_length=100,
                                                         init_token='&',
                                                         pad_token='&',
                                                         tokenize=self.tokenize_label,
                                                         batch_first=True)

    def gen_dataloader(self, train_file):
        smile_data_training = torchtext.data.TabularDataset(path=train_file,
                                                  format='csv',
                                                  fields=[('input', self.smile_mol_tokenizer),
                                                          ('output', self.smile_mol_masked_tokenizer)])

        self.train_data, self.test_data = smile_data_training.split(split_ratio=0.8)

        self.smile_mol_tokenizer.build_vocab(smile_data_training)
        self.smile_mol_masked_tokenizer.vocab = self.smile_mol_tokenizer.vocab
        self.train_batch, self.test_batch = torchtext.data.BucketIterator.splits((self.train_data, self.test_data),
                                                                      batch_size=512,
                                                                      shuffle=True,
                                                                      device=self.device,
                                                                      repeat=False,
                                                                      sort=False)

    def build_model(self):
        counts = self.smile_mol_tokenizer.vocab.freqs
        sum_counts = 0
        for tok, cnt in counts.items():
            if not tok in self.untargeted_tokens:
                sum_counts += cnt

        if self.class_weight == 'none':
            class_weights = [1.0 for x in range(self.n_tokens)]
        else:
            class_weights = [0., 0., 0.] ## <unk>, <PAD>, <REP>
            counts_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            for tok, cnt in counts_sorted:
                if not tok in self.untargeted_tokens:
                    if self.class_weight == 'log':
                        class_weights.append(math.log(sum_counts/cnt)/(self.n_tokens-5))
                    elif self.class_weight == 'sqrt':
                        class_weights.append(math.sqrt(sum_counts/cnt)/(self.n_tokens-5))
                    elif self.class_weight == 'raw':
                        class_weights.append((sum_counts/cnt)/(self.n_tokens-5))
                else:
                    class_weights.append(0.)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.ignored_token,
                                                   weight=torch.Tensor(class_weights).to(self.device))
        self.model = MoleculeTransformer(self.n_tokens,
                                      self.mol_emsize,
                                      self.n_head,
                                      self.n_hid,
                                      self.n_layers).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.decay_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1., gamma=0.99)

    def train(self, log='stdout'):
        if log == 'fout':
            log_file = open('train_log.txt', 'w')

        self.model.train()
        total_loss = 0.
        start_time = time.time()
        i = 0
        for batch in self.train_batch:
            data, targets = batch.input, batch.output
            self.optimizer.zero_grad()

            predicts = self.model(data).transpose(0, 1)
            loss = self.criterion(predicts.reshape(-1, self.n_tokens), targets.view(-1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = 200
            schedule_interval = 10000

            if i % log_interval == 0 and i > 0:
                # ACC check
                predicted_val = torch.max(torch.softmax(predicts, 2), 2)[1]
                masked_num, masked_hit = self.count_acc(batch.input, predicted_val, batch.output, self.mask_token)
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                log_str = ' {:5d}/{:5d} batches | lr {:02.10f} | ms/batch {:5.2f} | loss {:5.8f} | acc {:6.4f}'.format(
                            i, len(self.train_batch), self.decay_scheduler.get_last_lr()[0],
                            elapsed * 1000 / log_interval, cur_loss, masked_hit/masked_num)
                if log == 'stdout':
                    print(log_str)
                elif log == 'fout':
                    log_file.write(log_str + "\n")
                elif log == 'none':
                    pass
                total_loss = 0
                start_time = time.time()

                self.acc_history.append(masked_hit/masked_num)
            self.loss_history.append(loss)

            if i % schedule_interval == 0 and i > 0:
                self.decay_scheduler.step()
            i += 1

    def evaluate(self):
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for batch in self.test_batch:
                data, targets = batch.input, batch.output
                predicts = self.model(data).transpose(0,1)
                total_loss += len(data) * self.criterion(predicts.reshape(-1, self.n_tokens), targets.view(-1)).item()

        return total_loss / (len(test_batch) - 1)

    def evaluate_acc(self):
        self.model.eval()
        total_masked_num = 0
        total_masked_hit = 0
        with torch.no_grad():
            for batch in self.test_batch:
                data, targets = batch.input, batch.output
                predicts = self.model(data).transpose(0, 1)
                predicted_val = torch.max(torch.softmax(predicts, 2), 2)[1]
                masked_num, masked_hit = self.count_acc(batch.input, predicted_val, batch.output, self.mask_token)
                total_masked_num += masked_num
                total_masked_hit += masked_hit
        return total_masked_hit/total_masked_num

    def export_training_figure(self, epochs=1):
        plt.figure(figsize=(14,10))
        plt.plot(self.loss_history)
        plt.title("The history of training losses")
        plt.savefig("training_loss_{}.png".format(epochs))

        plt.figure(figsize=(14,10))
        plt.plot(self.acc_history)
        plt.title("The history of training accuracy")
        plt.savefig("training_accuracy_{}.png".format(epochs))

    def save_model(self, PATH="."):
        """
        """
        torch.save(self.model.state_dict(), PATH)

    def load_model(self, PATH="."):
        self.model = MoleculeTransformer(self.n_tokens,
                                      self.mol_emsize,
                                      self.n_head,
                                      self.n_hid,
                                      self.n_layers).to(self.device)
        self.model.load_state_dict(torch.load(PATH))

    def print_params(self):
        print("Model parameters::")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

    def print_status(self):
        print("Model hyperparameters::")
        pass

    @staticmethod
    def count_hit(h, Y, idx):
        """
        h : (N, S) predicted results from transformer should be transposed.
        Y : (N, S) ground truths of molecule tokens.
        idx : The token index of <unk>, which is the masked positions of labels.

        N = Batch size
        S = Sequence length

        return:
        """
        masked_num = (Y != idx).sum()
        masked_hit = torch.logical_and(Y - h == 0, h != idx).sum()

        return float(masked_num), float(masked_hit)

    @staticmethod
    def count_acc(X, h, Y, idx):
        """
        X : (N, S) input data of the transformer model.
        h : (N, S) predicted results from transformer should be transposed.
        Y : (N, S) ground truths of molecule tokens.
        idx : The token index of the masked token.

        N = Batch size
        S = Sequence length

        return:
        """
        masked_pos = (X == idx)
        masked_num = masked_pos.sum()
        masked_hit = torch.logical_and(Y - h == 0, masked_pos).sum()

        return float(masked_num), float(masked_hit)

    @staticmethod
    def tokenize_label(mol_str):
        """
        Tokenize function for labeled molecule data.

        """
        mol_str_list = list(mol_str)
        mol_str_list[0] = '&'
        if mol_str_list[-1] == '$':
            mol_str_list[-1] = '&'
        return mol_str_list

    @staticmethod
    def tokenize_train_new(mol_str):
        """
        Experimental tokenize function for training sets.
        1. Two letters organics are going to be a single token.
            e.g) Br, Cl
        2. Chemical with brackets are going to be a single token.
            e.g) [NH4+]

        return: tokenized list
        """
        tokens = []
        long_chr = False
        mol_str = mol_str.rstrip()
        current_str = ''

        i = 0
        while i < len(mol_str):
            if mol_str[i] == '[':
                long_chr = True
            elif mol_str[i] == ']':
                long_chr = False
            else:
                current_str += mol_str[i]

            if mol_str[i] == 'B': # B could be Br.
                if i+1 < len(mol_str) and mol_str[i+1] == 'r':
                    current_str += mol_str[i+1]
                    i += 1

            if mol_str[i] == 'C': # C could be Cl.
                if i+1 < len(mol_str) and mol_str[i+1] == 'l':
                    current_str += mol_str[i+1]
                    i += 1

            if not long_chr:
                tokens.append(current_str)
                current_str = ''
            i += 1
        return tokens
