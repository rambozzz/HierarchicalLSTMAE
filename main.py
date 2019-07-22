import torch.nn as nn
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import get_loader
from HierLSTM import HierVAE
from utils import init_weights, to_var, epoch_time
import os
import sys
import time
import gc
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

LOAD_CHECKPOINT = False


def main():

    vocab_file = open(root_path + "/data/dictionary", "r")
    docs_file = open(root_path + "/data/train_target_permute_segment.txt", "r")

    #truncate_ratio MUST BE a number "n" so that: 0<n<=1 . This is the ratio of the whole dataset to include
    train_loader, eval_loader, vocab_size, pad_idx, sod_idx = get_loader(docs_file.readlines(), vocab_file.readlines(), batch_size=BATCH_SIZE, shuffle=True, num_workers=12, max_doc_len=10, max_sent_len=15, truncate_ratio=0.6)

    vocab_file.close()
    docs_file.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("******DEVICE: "+ str(device))

    print("***Creating and initializing Model***")
    model = HierVAE(vocab_size, embedding_size=EMB_DIM, hidden_size=HID_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, device=device, sod_idx=sod_idx).to(device)


    if LOAD_CHECKPOINT:
        model.load_state_dict(torch.load("tut1-model-checkpoint.pt"))
    else:
        model.apply(init_weights)

    params = list(model.parameters())

    #optimizer = optim.Adam(params, lr=LR)
    optimizer = optim.SGD(params, lr=LR, momentum=1)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP, epoch, N_EPOCHS, LR)
        valid_loss = evaluate(model, eval_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')




def train(model, dataloader, optimizer, criterion, clip, epoch, num_epoch, LR):

    model.train()

    epoch_loss = 0

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    total_steps = len(dataloader)

    partial_loss = None

    reduce_lr = 0

    print("***Starting training...***")

    for i, (src, trg) in enumerate(dataloader):

        optimizer.zero_grad()

        trg, _ = rnn.pad_packed_sequence(trg, batch_first=True)
        src, lenghts = rnn.pad_packed_sequence(src, batch_first=True)

        src = to_var(src)
        trg = to_var(trg)



        output = model(src)

        # trg = [batch size, doc len, trg sent len]
        # output = [batch size, doc len, src_sent_len, output dim]

        output = output[:, :, 1:].contiguous().view(-1, output.shape[-1])
        trg = trg[:, :, 1:].contiguous().view(-1)

        # trg = [(doc len -1) * (trg sent len - 1) * batch size]
        # output = [(doc len -1) * (src sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)



        optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()

        epoch_loss += loss.detach().item()


        if i % log_step == 0 and i != 0:
            print('Epoch [%d/%d], Step [%d/%d], Partial current loss: %f'
                  % (epoch, num_epoch, i, total_steps, epoch_loss/i))


            print("Memory allocated: "+str(torch.cuda.memory_allocated() / 1024 ** 2)+" MB")

            '''
            if partial_loss == None:
                partial_loss = epoch_loss/i
            else:
                if partial_loss  < epoch_loss/i:
                    reduce_lr += 1
                    if reduce_lr == 20:
                        print("*****Reducing learning rate******")
                        reduce_lr = 0
                        LR = LR * 0.5
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = LR

                else:
                    partial_loss = epoch_loss/i
                    reduce_lr = 0
            '''

        if i%int(len(dataloader)/3) == 0 and i != 0:
            print("*****Reducing learning rate******")
            reduce_lr = 0
            LR = LR * 0.6
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR

        if i % save_step == 0 and i != 0:
            print("**********Saving checkpoint at current step************")
            torch.save(model.state_dict(), 'tut1-model-checkpoint_prova.pt')



    return epoch_loss / total_steps



def evaluate(model, dataloader, criterion):
    model.eval()

    epoch_loss = 0

    print("***Starting evaluation***")

    for i, (src, trg) in enumerate(dataloader):

        trg, _ = rnn.pad_packed_sequence(trg, batch_first=True)
        src, _ = rnn.pad_packed_sequence(src, batch_first=True)

        src = to_var(src)
        trg = to_var(trg)

        output = model(src, 0)

        # trg = [batch size, doc len, trg sent len]
        # output = [batch size, doc len, src_sent_len, output dim]

        output = output[:, 1:, 1:].contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:, 1:].contiguous().view(-1)

        # trg = [(doc len -1) * (trg sent len - 1) * batch size]
        # output = [(doc len -1) * (src sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)





####################################################
##################### MAIN #########################
####################################################
main()