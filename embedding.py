import torch.nn as nn
import torch
import sys
import os

master_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
file = open(master_dir+"/data/dictionary", "r")
lines = file.readlines()

vocab_size = len(lines)
vector_size = 16

embed = nn.Embedding(vocab_size, vector_size)


word_to_ix = {word.split("\n")[0]: i for i, word in enumerate(lines) }
word_indexes = torch.tensor([word_to_ix[w] for w in word_to_ix], dtype = torch.long)
print (word_indexes)
#word_vectors = embed(word_indexes)

for vector in word_indexes:
    print (embed(vector))


