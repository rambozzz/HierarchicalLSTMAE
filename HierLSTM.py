import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from utils import to_var, repackage_hidden
import random



'''
ENCODER
'''


class Encoder_sent(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder_sent, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, sentence):

        #sentence = [batch_size, sent_len]

        emb = self.drop(self.embedding(sentence))

        #emb = [batch_size, sent_len, emb_dim]

        outputs, (hidden, cell) = self.lstm(emb)

        #outputs = [batch size, sent len, hid dim]
        #hidden = [n_layers, batch_size, hid_dim]
        #cell = [n_layers, batch_size, hid_dim]
        #hidden = hidden[-1, :, :]
        return outputs[:, -1]#hidden



class Encoder_doc(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, enc_sentences):

        _, (hidden, cell) = self.lstm(self.drop(enc_sentences))
        #outputs, (hidden, cell) = self.lstm(self.drop(enc_sentences))

        return hidden, cell



class HierENC(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, device):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.enc_sent = Encoder_sent(vocab_size, embedding_size, hidden_size, num_layers, dropout)
        self.enc_doc = Encoder_doc(hidden_size, num_layers, dropout)

    def forward(self, doc):

        batch_size = doc.shape[0]
        n_sentences = doc.shape[1]

        enc_sentences = torch.zeros(batch_size, n_sentences, self.hidden_size).to(self.device)

        for i in range(0, n_sentences):

            enc_sentences[:, i, :] = self.enc_sent(doc[:, i, :])

        #enc_sentences = [batch_size, n_sentences, hid_dim]

        hidden_doc, cell_doc = self.enc_doc(enc_sentences)


        #hidden_doc = [n_layers, batch_size, hid_dim]
        #cell_doc = [n_layers, batch_size, hid_dim]

        return hidden_doc, cell_doc


'''
DECODER
'''



class Decoder_doc(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, input, hidden_doc, cell_doc):

        #input = self.drop(self.embedding(input)).clone()
        input = self.drop(input).clone()

        output, (hidden, cell) = self.lstm(input, (hidden_doc, cell_doc))

        return output.clone(), hidden.clone(), cell.clone()



class Decoder_sent(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.lin = nn.Linear(hid_dim, output_dim)

        #self.out = nn.Softmax(1)


        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input)).permute(1, 0, 2)

        # embedded = [batch size, 1, emb dim]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # output = [batch size, sent len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [batch size, 1, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        #prediction = self.out(self.lin(output.squeeze(1)))
        prediction = self.lin(output.squeeze(1))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


'''
HIER-VAE
'''

class HierVAE(nn.Module):
    def __init__(self,vocab_size, embedding_size, hidden_size, num_layers, dropout, device, sod_idx):
        super().__init__()
        self.device = device
        self.sod_idx = sod_idx
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hier_enc = HierENC(vocab_size, embedding_size, hidden_size, num_layers, dropout, device)
        self.dec_doc = Decoder_doc(vocab_size, embedding_size, hidden_size, num_layers, dropout)
        self.dec_sent = Decoder_sent(vocab_size, embedding_size, hidden_size, num_layers, dropout)

    def forward(self,src, teacher_forcing_ratio = 0.5):

        batch_size = src.shape[0]
        max_doc_len = src.shape[1]
        max_sent_len = src.shape[2]
        embedder = self.hier_enc.enc_sent.embedding

        output_doc = torch.zeros(batch_size, max_doc_len, max_sent_len, self.vocab_size).to(self.device)

        hidden_doc, cell_doc = self.hier_enc(src)
        #hidden_doc = self.hier_enc(src)


        input_sent = embedder(src[:, 0, 0])#<SOD> token, first element of the first sentence of each document
        input_sent = input_sent.unsqueeze(0).permute(1, 0, 2)
        #input_sent = [batch_size]

        for sent_idx in range(0, max_doc_len):
            sent_out, hidden_sent, cell_sent = self.dec_doc(input_sent, hidden_doc, cell_doc)
            input_sent = sent_out
            if sent_idx == 0:
                input_word = src[:, 0, 0] #<SOS> token, first e
            else:
                input_word = src[:, 1, 0]


            output_sent = torch.zeros(batch_size,max_sent_len, self.vocab_size).to(self.device)
            for word_idx in range(1, max_sent_len):
                prediction, hidden_sent, cell_sent = self.dec_sent(input_word, hidden_sent, cell_sent)
                output_sent[:, word_idx] = prediction
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = prediction.max(1)[1]
                input_word = (src[:, sent_idx, word_idx] if teacher_force else top1)
            hidden_doc, cell_doc = hidden_sent, cell_sent
            output_doc[:, sent_idx] = output_sent

        return output_doc
