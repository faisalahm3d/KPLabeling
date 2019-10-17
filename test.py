import re

import os
import codecs
import sys
import logging
import glob
import json
import numpy as np
'''
with open("keyword_kdd", 'r') as tt:
    keyphrases = json.load(tt)

with open("documents.json", 'r') as doc_file:
    documents = json.load(doc_file)
doc_key = "3744272"
'''

with open("tag_documents.json", 'r') as doc_file:
    tag_documents = json.load(doc_file)

#doc_value = documents[doc_key]
''''
for line in doc_value[3:]:
    temp_line = line
    for kp in keyphrases[doc_key]:
        res = [m.start() for m in re.finditer(kp, temp_line)]
        count = 0
        for i in res:
            offset = count + i
            temp1 = temp_line[0:offset]
            length = len(kp)
            temp2 = temp_line[offset + length:]
            split_kp = kp.split()
            l = len(split_kp)
            tag = [value + '_B-KP' if index is 0 else value + '_I-KP' for index, value in enumerate(split_kp)]
            temp3 = ' '.join(tag)
            new_line = temp1 + temp3 + temp2
            count = count + l* 5
            temp_line = new_line

    print(line)
    print('----------------')
    print(temp_line)
    print('-----------------------------')
    print('-----------------------------')
'''



tag_tokens = []
mis_tag = []
right_tag = []
for do_key, do_value in tag_documents.items():
    doc = ' '.join(do_value)
    tokens = doc.strip().split()
    #word_kp_pair = [(token.split('_')[0], token.split('_')[1]) if len(token.split('_'))==2 else (token.split('_')[0],'O') for token in tokens]
    tag_test = [token.split('_')[1] if len(token.split('_'))==2 else 'O' for token in tokens]
    #print(word_kp_pair
    ttt = set(tag_test)

    for t in ttt:
        if t=='O' or t=='B-KP' or t == 'I-KP':
            right_tag.append(do_key)
        else:
            mis_tag.append(do_key)
    #tag_tokens.append(word_kp_pair)


'''
with open("tag_corpus.json", 'w') as o:
    json.dump(tag_tokens, o, sort_keys = True, indent = 4)
    

sentences, sentence_tags =[], []
for tagged_sentence in tag_tokens:
    sentence, tags = zip(*tagged_sentence)
    sentences.append(np.array(sentence))
    sentence_tags.append(np.array(tags))
print(sentences[5])
print(sentence_tags[5])

'''
print(set(mis_tag))
print (len(set(right_tag)))
for i in set(mis_tag):
    print(tag_documents[i])






