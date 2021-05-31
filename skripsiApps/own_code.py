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


def sentence_similarity(sentence1, sentence2 , fasttextModel):
  sentence1 = ' '.join(sentence1)
  sentence2 = ' '.join(sentence2)

  vector1 = fasttextModel.get_sentence_vector(sentence1)
  vector2 = fasttextModel.get_sentence_vector(sentence2)

  xy= 0
  x = 0
  y = 0
  for i in range(len(vector2)):
    xy += vector1[i] * vector2[i]
    x += vector1[i]**2
    y += vector2[i]**2
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

def MMR(textrank_dict, sentence_target, model, mmr_lambda) : 
  n = sentence_target #for apps purpose
  #set hyper parameter
  #alpha value is between 0 - 1 
  print(mmr_lambda)
  summarySet = []
  while n > 0 : 
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
          mmr[sentence] = mmr_lambda * textrank_dict[sentence] - (1-mmr_lambda) * sentence_similarity(sentence, SummaryInOneSentences , model)
    selected = max(mmr.items(), key = operator.itemgetter(1))[0]
    summarySet.append(selected)
    n -=1
  return summarySet

def summarize(text, percentage, mmr_lambda, model):
  original_sentences = sent_tokenize(text) #to be constructed later on final summary
  words = preprocess(text)
  percentage = int(percentage)

  textrank_summary_dict = textrank(words,original_sentences, percentage, model)
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

def bulk_summarize(text,type, percentage_original , mmr_lambda, model):
    result = {}
    result['text'] = text
    result['summary'] = []

    for i in range(len(text)):
        original_sentences = sent_tokenize(text[i])
        if(type != "fixed"):
            percentage = int(percentage_original * len(original_sentences) / 100 )
            if (percentage >= len(original_sentences)):
              percentage = len(original_sentences)
            if (percentage == 0):
              percentage = 1
        else :
            percentage = int(percentage_original)


        print("percentage  : " , percentage)
        print("orignial :  " , len(original_sentences))

        if percentage <= len(original_sentences) : 
              words = preprocess(text[i])
              textrank_summary_dict = textrank(words,original_sentences, percentage, model)
              mmr_summary_list = MMR(textrank_summary_dict, percentage, model, mmr_lambda)

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