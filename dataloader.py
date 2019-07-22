
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
import math



class TextDataset(data.Dataset):
    def __init__(self, lines, vocabulary_lines, max_doc_len, max_sent_len, truncate_ratio):
        self.max_doc_len = max_doc_len
        self.max_sent_len = max_sent_len
        self.truncate_ratio = truncate_ratio
        self.word_to_ix, self.vocab_size, self.ix_to_word = self.init_w2ix(vocabulary_lines)
        print (self.word_to_ix)
        self.word_tensors = torch.tensor([self.word_to_ix[w] for w in self.word_to_ix], dtype = torch.long)
        self.src, self.trg = self.define_docs(lines, self.truncate_ratio)


    def init_w2ix(self, vocab_lines):
        word_to_ix = {word.split("\n")[0]: i+1 for i, word in enumerate(vocab_lines)}
        word_to_ix["<PAD>"] = 0
        word_to_ix["<EOS>"] = len(vocab_lines) + 1
        word_to_ix["<SOS>"] = len(vocab_lines) + 2
        word_to_ix["<SOD>"] = len(vocab_lines) + 3
        word_to_ix["<EOD>"] = len(vocab_lines) + 4
        vocab_size = len(word_to_ix)

        ix_to_word = {ix: word for word, ix in word_to_ix.items()}

        return word_to_ix, vocab_size, ix_to_word

    def define_docs(self, lines, truncate_ratio):

        #Discards documents with more than "self.max_doc_len" sentences per document OR documents containing sentences with more than "self.max_sent_len" words
        src_docs = []
        trg_docs = []
        src_doc = []
        trg_doc = []
        max_sent_len  = 0
        for line in lines:
            if line != "\n":
                src = line.replace("\n", "").split()
                if len(src) > max_sent_len:
                    max_sent_len = len(src)

                trg = src[::-1]

                src.insert(0, self.word_to_ix["<SOS>"])
                trg.insert(0, self.word_to_ix["<SOS>"])
                src.insert(len(src), self.word_to_ix["<EOS>"])
                trg.insert(len(src), self.word_to_ix["<EOS>"])
                src_doc.append(src)
                trg_doc.append(trg)
            else:
                doc_len = len(src_doc)
                if not (doc_len > self.max_doc_len or max_sent_len > self.max_sent_len):
                    trg_doc.reverse()
                    #src_doc[0].insert(0, self.word_to_ix["<SOD>"])
                    src_doc[0][0] = self.word_to_ix["<SOD>"]
                    trg_doc[0][0] = self.word_to_ix["<SOD>"]
                    #trg_doc[0].insert(0, self.word_to_ix["<SOD>"])
                    src_doc[-1].insert(-1, self.word_to_ix["<EOD>"])
                    trg_doc[-1].insert(-1, self.word_to_ix["<EOD>"])


                    #src_doc.insert(0, self.word_to_ix["<SOD>"])
                    #trg_doc.insert(0, self.word_to_ix["<SOD>"])
                    #src_doc.insert(len(src_doc), self.word_to_ix["<EOD>"])
                    #trg_doc.insert(len(trg_doc), self.word_to_ix["<EOD>"])
                    src_docs.append(src_doc)
                    trg_docs.append(trg_doc)

                src_doc = []
                trg_doc = []
                max_sent_len = 0

                dataset_truncate = int(len(src_docs)*truncate_ratio)

        return src_docs[:dataset_truncate], trg_docs[:dataset_truncate]

    def get_vocab_size(self):
        return self.vocab_size

    def get_pad_idx(self):
        return self.word_to_ix["<PAD>"]

    def get_sod_idx(self):
        return self.word_to_ix["<SOD>"]


    def __getitem__(self, index):
        #return self.docs[index]
        curr_src = self.src[index]
        curr_trg = self.trg[index]

        max_len = 0
        for i, line in enumerate(curr_src):
            int_src = self.list_sent(line)
            int_trg = self.list_sent(curr_trg[i])
            if len(int_src) > max_len:
                max_len = len(int_src)
            curr_src[i] = int_src
            curr_trg[i] = int_trg
        #while len(tens_doc) < self.max_doc_len:
            #tens_doc.append(torch.zeros([self.max_sent_len], dtype=torch.long))
        #padded_doc = torch.nn.utils.rnn.pad_sequence(tens_doc, batch_first=True)
        return (curr_src,curr_trg,max_len, len(curr_src))

    '''
    def pad_sent(self, sent):
        sent = [int(w) for w in sent]
        while len(sent) < self.max_sent_len:
            sent.append(0)
        return sent
    '''

    def list_sent(self, sent):
        if isinstance(sent, int):
            return [sent]
        sent = [int(w) for w in sent]
        return sent


    def __len__(self):
        return len(self.src)

'''
def pack_batch(batch):

    batch = rnn.pack_sequence(batch, enforce_sorted=False)
    #padded_batch = rnn.pad_packed_sequence(batch, batch_first=True)
    return batch

'''
def pack_batch(batch):

    max_sent_len = max([tuple[2] for tuple in batch ])
    max_doc_len = max([tuple[3] for tuple in batch ])
    #src_tens = torch.zeros(len(batch), max_doc_len, max_sent_len)
    #trg_tens = torch.zeros(len(batch), max_doc_len, max_sent_len)
    src_batch = []
    trg_batch = []
    for j,(src_doc, trg_doc, _, _) in enumerate(batch):
        for i, line in enumerate(src_doc):
            while len(line) < max_sent_len:
                line.append(0)
            src_doc[i] = torch.tensor(line, dtype=torch.long)
            while len(trg_doc[i]) < max_sent_len:
                trg_doc[i].append(0)
            trg_doc[i] = torch.tensor(trg_doc[i], dtype=torch.long)
        #src_tens[j, :, :] = torch.stack(src_doc)
        #trg_tens[j, :, :] = torch.stack(trg_doc)

        src_batch.append(torch.stack(src_doc))
        trg_batch.append(torch.stack(trg_doc))

    src_batch = rnn.pack_sequence(src_batch, enforce_sorted=False)
    trg_batch = rnn.pack_sequence(trg_batch, enforce_sorted=False)

    return (src_batch, trg_batch)



def get_loader(lines, vocabulary_lines, batch_size, shuffle, num_workers, max_doc_len, max_sent_len, truncate_ratio):
    dataset = TextDataset(lines=lines, vocabulary_lines=vocabulary_lines, max_doc_len=max_doc_len, max_sent_len=max_sent_len, truncate_ratio=truncate_ratio)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print("***Splitting dataset for train and val***")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("***TRAIN_SET LENGTH: [%d], VAL_SET LENGTH: [%d]" %(len(train_dataset), len(test_dataset)))

    vocab_size = dataset.get_vocab_size()
    pad_idx = dataset.get_pad_idx()
    sod_idx = dataset.get_sod_idx()

    return torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pack_batch), torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                collate_fn=pack_batch), vocab_size, pad_idx, sod_idx

def get_TEST_loader(lines, vocabulary_lines, batch_size, shuffle, num_workers, max_doc_len, max_sent_len, truncate_ratio):
    dataset = TextDataset(lines=lines, vocabulary_lines=vocabulary_lines, max_doc_len=max_doc_len, max_sent_len=max_sent_len, truncate_ratio=truncate_ratio)
    vocab_size = dataset.get_vocab_size()
    pad_idx = dataset.get_pad_idx()
    sod_idx = dataset.get_sod_idx()

    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pack_batch), vocab_size, pad_idx, sod_idx