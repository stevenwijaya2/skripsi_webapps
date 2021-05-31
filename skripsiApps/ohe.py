import nltk
import pandas as pd
import numpy as np
import operator 
import networkx as nx
import warnings

warnings.filterwarnings("ignore")
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from statistics import mean
import math
nltk.download('punkt')


def preprocess(text) : 
  #sentences tokenizing
  sentences = sent_tokenize(text)

  #case folding
  sentences = [word.lower() for word in sentences]

  # Stemming
  stemmed = []
  stemmer = StemmerFactory().create_stemmer()
  for i in range(len(sentences)):
    stemmed.append(stemmer.stem(sentences[i]))

  #filtering (stopword removal)
  filtered = []
  # stopwordList = StopWordRemoverFactory().get_stop_words()
  stopwordRemoval = StopWordRemoverFactory().create_stop_word_remover()
  for i in range(len(stemmed)) : 
    filtered.append(stopwordRemoval.remove(stemmed[i]))

  #word tokenizing
  words = []
  for i in range(len(filtered)):
    words.append(word_tokenize(filtered[i]))
  return words

def build_similarity_matrix(sentences , fasttextModel):
  S = np.ones((len(sentences),len (sentences)))

  for i in range(len(sentences)):
    for j in range(len(sentences)):
      if i == j : 
        continue
      sentence1 = ' '.join(sentences[i])
      sentence2 = ' '.join(sentences[j])
      S[i][j] = sentence_similarity(sentence1 , sentence2 , fasttextModel)
  for k in range(len(S)) : 
    S[k] /= S[k].sum()
  return S  


def sentence_similarity(sentence1, sentence2 , all_words):
  tokenize_sentence1 = word_tokenize(sentence1)
  tokenize_sentence2 = word_tokenize(sentence2) 

  vector1 = [0] * len(all_words)
  vector2 = [0] * len(all_words)

  for word in tokenize_sentence1:
    vector1[all_words.index(word)] += 1
  for word in tokenize_sentence2:
    vector2[all_words.index(word)] += 1
  xy= 0
  x = 0
  y = 0
  for i in range(len(all_words)):
    xy += vector1[i] * vector2[i]
    x += vector1[i]**2
    y += vector2[i]**2
  if (x == 0) or (y == 0) or (xy == 0):
    return 0
  similarity = xy / ((x**(1.0 / 2)) * (y**(1.0 / 2)))
  return similarity

def textrank(words, original_sentences , sentence_target , model):
    # create similarity matrix
    similarity_matrix = build_similarity_matrix(words, model)

    # applying pagerank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    pagerank_sentences = nx.pagerank_numpy(nx_graph)
    ranked_pagerank = sorted(pagerank_sentences.items(), key=lambda kv: kv[1], reverse=True)

    # create the sentences order , for appending the original sentences later
    sentence_order = []
    for x in range(len(ranked_pagerank)):
        sentence_order.append(ranked_pagerank[x][0])

    # append the sentences
    ranked_sentences = []
    for i in range(len(words)):
      ranked_sentences.append(' '.join(words[sentence_order[i]]))
    
    #append the scores to list
    ranked_scores = []
    for x in range(len(ranked_pagerank)):
        ranked_scores.append(ranked_pagerank[x][1])

#   reduce sentences on textrank summary
    n = sentence_target #for apps purpose
    lim = len(original_sentences)
    pagerank_reduced = ranked_sentences.copy()
    while lim > n:
        if ranked_scores[lim-1] > mean(ranked_scores) : 
            break
        pagerank_reduced.pop()
        lim -= 1

    # create dict for later used on MMR
    ranked_dict = {}
    for i in range(len(pagerank_reduced)):
        ranked_dict[pagerank_reduced[i]] = ranked_pagerank[i][1]
    sorted_keys = sorted(ranked_dict, key=ranked_dict.get, reverse=True)

    # append the score to dict
    sorted_dict = {}
    for w in sorted_keys:
        sorted_dict[w] = ranked_dict[w]
    return sorted_dict

def MMR(textrank_dict, sentence_target, model , mmr_lambda) : 
  n = sentence_target #for apps purpose
  #set hyper parameter
  #alpha value is between 0 - 1 
  alpha = mmr_lambda
  summarySet = []
  while math.floor(n) > 0 : 
    mmr = {}
    for sentence in textrank_dict.keys():
      if not sentence in summarySet:
        #set the textrank score as initial value
        if (len(summarySet) == 0) : 
            mmr[sentence] = textrank_dict[sentence]
        else : 
          SummaryInOneSentences = ''
          for i in range(len(summarySet)):
            SummaryInOneSentences += (summarySet[i] + " ")
          mmr[sentence] = alpha * textrank_dict[sentence] - (1-alpha) * sentence_similarity(sentence, SummaryInOneSentences , model)
    selected = max(mmr.items(), key = operator.itemgetter(1))[0]
    summarySet.append(selected)
    n -=1
  return summarySet

def summarize(text, percentage, mmr_lambda):

  original_sentences = sent_tokenize(text)
  percentage = int(percentage)
  # preprocess news article
  words = preprocess(text)

  # join every item from preprocessed list and create list 1 dimension
  model = np.concatenate(words)
  model = model.tolist()

  # reduce redundant word from word dictionary
  list_of_unique_word = []
  unique_word = set(model)
  for word in unique_word:
    list_of_unique_word.append(word)
  model = list_of_unique_word.copy()


  textrank_summary_dict = textrank(words,original_sentences, percentage,model)
  mmr_summary_list = MMR(textrank_summary_dict, percentage, model, mmr_lambda)

  sentences_order = []
  for i in mmr_summary_list:
    sentences_order.append(words.index(word_tokenize(i)))
  
  #append the original sentences
  final_summary = []
  for j in sentences_order:
    final_summary.append(original_sentences[j])
  final_summary_string = ' '.join(final_summary) 
  return final_summary_string

def bulk_summarize(text,type, percentage_original , mmr_lambda):
    result = {}
    result['text'] = text
    result['summary'] = []

    for i in range(len(text)):
        original_sentences = sent_tokenize(text[i])
        if(type != "fixed"):
            percentage = int(percentage_original * len(original_sentences) / 100 )
        else :
            percentage = int(percentage_original)

        print(i , " >> " , percentage)
        if percentage < len(original_sentences) : 
            # preprocess news article
            words = preprocess(text[i])

            # join every item from preprocessed list and create list 1 dimension
            model = np.concatenate(words)
            model = model.tolist()

            # reduce redundant word from word dictionary
            list_of_unique_word = []
            unique_word = set(model)
            for word in unique_word:
                list_of_unique_word.append(word)
            model = list_of_unique_word.copy()


            textrank_summary_dict = textrank(words,original_sentences, percentage,model)
            mmr_summary_list = MMR(textrank_summary_dict, percentage,model , mmr_lambda)

            sentences_order = []
            for i in mmr_summary_list:
                sentences_order.append(words.index(word_tokenize(i)))
            
            #append the original sentences
            final_summary = []
            for j in sentences_order:
                final_summary.append(original_sentences[j])
            final_summary_string = ' '.join(final_summary) 
            result['summary'].append(final_summary_string)
        else : 
            result['summary'].append("Error : Target summary Length is greater than you news length")
    df = pd.DataFrame(result)

    return df