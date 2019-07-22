import torch.nn as nn
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim
import gc

from dataloader import get_TEST_loader
from HierLSTM import HierVAE
from utils import init_weights, to_var, epoch_time
import os
import sys
import time
import math


root_path = master_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

EMB_DIM = 1000
HID_DIM = 1000
DROPOUT = 0.2
NUM_LAYERS = 4

BATCH_SIZE = 10
N_EPOCHS = 10
CLIP = 1
LR = 1e-1

log_step = 10
save_step = 1500

def main():

    vocab_file = open(root_path + "/data/dictionary", "r")
    docs_file = open(root_path + "/data/train_target_permute_segment.txt", "r")

    lines = docs_file.readlines()
    test_loader, vocab_size, PAD_IDX = get_TEST_loader(lines, vocab_file.readlines(), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, max_doc_len=15, max_sent_len=40, truncate_ratio=0.1)
    lines = None

    vocab_file.close()
    docs_file.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("***Creating and initializing Model***")
    model = HierVAE(vocab_size, embedding_size=EMB_DIM, hidden_size=HID_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, device=device).to(device)

    model.load_state_dict(torch.load("tut1-model.pt"))

    model.eval()

    softmax = nn.LogSoftmax(3)

    results = open("results/results.txt", "a")
    trgs = open("results/targets.txt", "a")
    with torch.no_grad():
        for i, (src, trg) in enumerate(test_loader):

            trg, _ = rnn.pad_packed_sequence(trg, batch_first=True)
            src, _ = rnn.pad_packed_sequence(src, batch_first=True)

            src = to_var(src)
            trg = to_var(trg)

            output = model(src, 0)

            # trg = [batch size, doc len, trg sent len]
            # output = [batch size, doc len, src_sent_len, output dim]

            output = softmax(output)
            final = torch.zeros(output.shape[0], output.shape[1], output.shape[2])


            for i in range(0, output.shape[1]):
                for j in range(0, output.shape[2]):
                    final[:, i, j] = output[:, i, j].max(1)[1]

            ix2word(final, trg, test_loader.dataset.ix_to_word, results, trgs)




def ix2word(output, trg, ix2word, results, trgs):
    output = output.cpu().numpy()
    trg = trg.cpu().numpy()



    for z, doc in enumerate(output.tolist()):

        out_curr = []
        out_trg = []
        for i, row in enumerate(doc):
            out_sent = ""
            out_trg_sent = ""
            for j, ix in enumerate(row):
                out_sent = out_sent + ix2word[row[j]] + " "
                out_trg_sent = out_trg_sent + ix2word[trg[z][i][j]] + " "
            out_sent = out_sent + "\n"
            out_trg_sent = out_trg_sent + "\n"
            out_curr.append(out_sent)
            out_trg.append(out_trg_sent)
        out_curr.append("\n")
        out_trg.append("\n")
        results.writelines(out_curr)
        trgs.writelines(out_trg)
    return




main()