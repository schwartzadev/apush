import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import logging

import get_sections_from_textbook

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logging.info('running...')

prompt = 'impact of fugitive slaves in the civil war the life of black soldiers in the military'


# this is using the first section (section_number) from chapter five (chapter_url)
chapter_url = "http://www.americanyawp.com/text/05-the-american-revolution/"
section_number = 2 # one indexed

logging.info('getting sections from textbook')
sections = get_sections_from_textbook.get_chapter_sections(chapter_url)

sentences = sections[section_number - 1]['content']
logging.info('using section {0} from chapter url {1}'.format(section_number, chapter_url))

sentences = [sent_tokenize(s) for s in sentences]

sentences = [y for x in sentences for y in x] # flatten list

# Extract word vectors
logging.info('extracting word vectors')
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
logging.info('word vectors extracted')

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# lowercase
clean_sentences = [s.lower() for s in clean_sentences]

stop_words = stopwords.words('english')

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

logging.info('generating sentence vectors')
sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)

# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])


for i in range(len(sentences)):
    for j in range(len(sentences)):
         if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


nx_graph = nx.from_numpy_array(sim_mat)
logging.info('sorting generated sentences')
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# Extract top 10 sentences as the summary
top = []
for i in range(15):
    top.append(ranked_sentences[i][1])


print(sections[section_number - 1]['header'])
[print('* ', sentence, '\n') for sentence in top]


# create sentence vector for prompt
promptVector = sum([word_embeddings.get(w, np.zeros((100,))) for w in prompt.split()])/(len(prompt.split())+0.001)

# find most similar sentence to prompt
#for i in range(len(sentences)):
 #   for j in range(len(sentences)):
  #       if i != j:
   #         sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
