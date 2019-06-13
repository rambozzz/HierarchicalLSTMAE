import torch.nn as nn
import torch
import torch.utils.data as data
import os
import sys
import time


root_path = master_dir = os.path.dirname(os.path.abspath(sys.argv[0]))




class TextDataset(data.Dataset):
    def __init__(self, lines, vocabulary_lines):
        self.vocab_size = len(vocabulary_lines)
        self.word_to_ix = {word.split("\n")[0]: i for i, word in enumerate(vocabulary_lines)}
        self.word_tensors = torch.tensor([self.word_to_ix[w] for w in self.word_to_ix], dtype = torch.long)
        self.docs = self.define_docs(lines)

    def define_docs(self, lines):
        docs = []
        doc = []
        for line in lines:
            if line != "\n":
                #tensor_line = self.gen_tensor_sent(line)
                #doc.append(tensor_line)
                doc.append(line)

            else:
                docs.append(doc)
                doc = []
        return docs

    def gen_tensor_sent(self, sentence):
        tensor_sent = []
        sentence = sentence.split()
        for word in sentence:
            word = word.strip("\n")
            tensor_sent.append(self.word_tensors[int(word)])
        #print(tensor_sent)
        return tensor_sent

    def __getitem__(self, index):
        #return self.docs[index]
        curr_doc = self.docs[index]
        #print (curr_doc)
        tens_doc = [self.gen_tensor_sent(line) for line in curr_doc]
        return [tens_doc]


    def __len__(self):
        return len(self.docs)

def get_loader(lines, vocabulary_lines, batch_size, shuffle, num_workers):
    dataset = TextDataset(lines=lines, vocabulary_lines=vocabulary_lines)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda x: x)



def main():

    vocab_file = open(root_path + "/data/dictionary", "r")
    docs_file = open(root_path + "/data/new_train_target_permute_segment.txt", "r")

    data_loader = get_loader(docs_file.readlines(), vocab_file.readlines(), batch_size=32, shuffle=False, num_workers=8)

    vocab_file.close()
    docs_file.close()

    start_time = time.time()

    for i, batch in enumerate(data_loader):
        for element in batch:
            print (element)
    print("--- %s seconds ---" % (time.time() - start_time))




main()