
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
import numpy as np
import collections
import os
import pickle
from nltk import corpus
class SemCorEmb(word_embeddings):
    def _init_(self):
        self.word_embeddings = word_embeddings
        self.raw_sentences = semcor.sents()

def getContextEmb(sentence,center,window_size,embedding_dict,emb_size):
    # Input introductions
    # sentence: an array of tokens of untagged sentence. 
    # center: position of the center word
    # window_size: size of context window
    # embedding_Dict: trained embedding dictionary used to calculate context
    ################################################################
    start_pos = max([0,center-window_size])
    end_pos = min([len(sentence),(center+window_size)+1])
    context_tokens = sentence[start_pos:end_pos]
    output_embedding = np.zeros(emb_size)
    for word in context_tokens:
        try:
            output_embedding+=embedding_dict[word]
        except:
            output_embedding+=np.random.uniform(1,-1,emb_size)
    return output_embedding

def buildSemEmb(tagged_sents,emb_size,context_builder = getContextEmb):
    output_dict = collections.defaultdict(lambda: np.zeros(emb_size))
    for sentence in tagged_sents:
        #print(sentence)
        for idx,chunk in enumerate(sentence):
            if(type(chunk))==list:
                continue
            else:
                #Use try except handling since some of the label is broken
                try:
                    sense_index = chunk.label().synset().name()
                except:
                    continue
                context_emb = context_builder(sentence,idx,3,embedding_dict,emb_size)
                output_dict[sense_index]+=context_emb
    return output_dict


def load_glove_embeddings(glove_directory,emsize=50,voc_size=50000):
    #get directory name glove.6B or other training corpus size
    if glove_directory[-1] =='/':
        dirname = glove_directory.split('/')[-2]
    else:
        dirname = glove_directory.split('/')[-1]
    if emsize in [50,100,300]:
        f = open(os.path.join(glove_directory,'%s.%sd.txt'%(dirname,emsize)))
    else:
        print('Please select from 50, 100 or 300')
        return
    loaded_embeddings = collections.defaultdict()
    for i, line in enumerate(f):
        if i >= voc_size: 
            break
        s = line.split()
        loaded_embeddings[s[0]] = np.asarray(s[1:],dtype='float64')
    return loaded_embeddings