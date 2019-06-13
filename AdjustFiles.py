import os
import sys

master_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
file = open(master_dir+"/data/train_source_permute_segment.txt", "r")
lines = file.readlines()
new_lines = []
for line in lines:
    if line != "\n":
        new_line = ""
        words = line.split()
        for i, word in enumerate(words):
            word.strip("\n")
            word = str(int(word) - 1)
            if i == len(words) - 1:
                word += "\n"
                new_line += word
            else:
                new_line = new_line + word + " "
    else:
        new_line = line
    new_lines.append(new_line)

with open(master_dir+"/data/new_train_source_permute_segment.txt", "w") as file:
    file.writelines(new_lines)




file = open(master_dir+"/data/train_target_permute_segment.txt", "r")
lines = file.readlines()
new_lines = []
for line in lines:
    if line != "\n":
        new_line = ""
        words = line.split()
        for i, word in enumerate(words):
            word.strip("\n")
            word = str(int(word) - 1)
            if i == len(words) - 1:
                word += "\n"
                new_line += word
            else:
                new_line = new_line + word + " "
    else:
        new_line = line
    new_lines.append(new_line)

with open(master_dir+"/data/new_train_target_permute_segment.txt", "w") as file:
    file.writelines(new_lines)