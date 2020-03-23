#!/usr/bin/env python
# coding: utf-8

import logging
import os
import ast

import codecs
import collections
import string

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import fasttext
import gensim
import pymorphy2


FORMAT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.DEBUG,format=FORMAT)


def convert_word(word, to_vectors=False):
    if to_vectors:
        return '_'.join(word.split()).lower()
    return ' '.join(word.split('_')).upper()


def get_top(v, matrix, words=None, k=10):
    similarities = cosine_similarity(v.reshape(1, -1), matrix).reshape(-1)
    top_indices = similarities.argsort()[-k:][::-1]
    return [(similarities[ind], words[ind] if words else ind)
            for ind in top_indices
            ]


####################################################################################
######################## TRAIN DATASET UTILS #######################################
####################################################################################
def read_gold_dataset(filepath):
    assert (os.path.isfile(filepath))
    train_dataset = pd.read_csv(filepath, sep='\t')

    gold_synsetid2parents = collections.defaultdict(list)

    for i, row in train_dataset.iterrows():
        # SYNSET_ID	TEXT	PARENTS	PARENT_TEXTS
        # 104745-N	РИЕКА	['207-N', '242-N', '590-N', '136274-N', '144998-N']	['ГАВАНЬ, ПОРТ, ПОРТОПУНКТ, ПОРТОВЫЙ ГОРОД, ПОРТОВЫЙ ПУНКТ', 'ГОРОДСКАЯ СРЕДА, ГОРОД, ГОРОДОК, ГОРОДСКАЯ МЕСТНОСТЬ', 'НАСЕЛЕННЫЙ ПУНКТ, ПОСЕЛЕНИЕ', 'ГОРОД ЕВРОПЫ, ГОРОД В ЕВРОПЕ, ЕВРОПЕЙСКИЙ ГОРОД', 'ХОРВАТСКИЙ ГОРОД, ГОРОД ХОРВАТИИ, ГОРОД В ХОРВАТИИ']
        synset_id = row['SYNSET_ID']
        parents = sorted(ast.literal_eval(row['PARENTS']))
        gold_synsetid2parents[synset_id].extend(parents)

    return gold_synsetid2parents


def read_train_dataset(filepath, ruwordnet):
    assert (os.path.isfile(filepath))
    train_dataset = pd.read_csv(filepath, sep='\t')

    word2parents = collections.defaultdict(list)

    for i, row in train_dataset.iterrows():
        # SYNSET_ID	TEXT	PARENTS	PARENT_TEXTS
        # 104745-N	РИЕКА	['207-N', '242-N', '590-N', '136274-N', '144998-N']	['ГАВАНЬ, ПОРТ, ПОРТОПУНКТ, ПОРТОВЫЙ ГОРОД, ПОРТОВЫЙ ПУНКТ', 'ГОРОДСКАЯ СРЕДА, ГОРОД, ГОРОДОК, ГОРОДСКАЯ МЕСТНОСТЬ', 'НАСЕЛЕННЫЙ ПУНКТ, ПОСЕЛЕНИЕ', 'ГОРОД ЕВРОПЫ, ГОРОД В ЕВРОПЕ, ЕВРОПЕЙСКИЙ ГОРОД', 'ХОРВАТСКИЙ ГОРОД, ГОРОД ХОРВАТИИ, ГОРОД В ХОРВАТИИ']
        parents = sorted(ast.literal_eval(row['PARENTS']))
        for w in ruwordnet.get_synset_senses_list(row['SYNSET_ID']):
            w = convert_word(w).lower()
            word2parents[w].append(parents)

    return word2parents


def read_train_dataset_s(filepath):
    assert (os.path.isfile(filepath))
    train_dataset = pd.read_csv(filepath, sep='\t')

    synsetid2parents = collections.defaultdict(list)

    for i, row in train_dataset.iterrows():
        # SYNSET_ID	TEXT	PARENTS	PARENT_TEXTS
        # 104745-N	РИЕКА	['207-N', '242-N', '590-N', '136274-N', '144998-N']	['ГАВАНЬ, ПОРТ, ПОРТОПУНКТ, ПОРТОВЫЙ ГОРОД, ПОРТОВЫЙ ПУНКТ', 'ГОРОДСКАЯ СРЕДА, ГОРОД, ГОРОДОК, ГОРОДСКАЯ МЕСТНОСТЬ', 'НАСЕЛЕННЫЙ ПУНКТ, ПОСЕЛЕНИЕ', 'ГОРОД ЕВРОПЫ, ГОРОД В ЕВРОПЕ, ЕВРОПЕЙСКИЙ ГОРОД', 'ХОРВАТСКИЙ ГОРОД, ГОРОД ХОРВАТИИ, ГОРОД В ХОРВАТИИ']
        parents = sorted(ast.literal_eval(row['PARENTS']))
        synset_id = row['SYNSET_ID']
        synsetid2parents[synset_id].append(parents)

    return synsetid2parents


def write_preds(f, w, hypernyms, ruwordnet, senses_sep):
    for hypernym in hypernyms:
        word = convert_word(w)
        f.write(f"{word}\t{hypernym}\t{ruwordnet.get_synset_senses(hypernym, sep=senses_sep)}\n")
    
    
def save_to_file(words_with_hypernyms, output_path, ruwordnet, senses_sep=', '):
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            if hasattr(ruwordnet, 'get_synset_senses_list'):
                senses = ruwordnet.get_synset_senses_list(word)
                if len(senses) > 0: # word is synset_id
                    for sense in senses:
                        write_preds(f, sense, hypernyms, ruwordnet, senses_sep)
                else: # word is sense
                    write_preds(f, word, hypernyms, ruwordnet, senses_sep)
            else:
                write_preds(f, word, hypernyms, ruwordnet, senses_sep)
                
####################################################################################
################ OUT OF VOCAB HANDLE AND NORMALIZE UTILS ###########################
##### SENTENCES NORMALIZATION ALGO ########
####################################################################################
# https://ru.wikibooks.org/wiki/%D0%A0%D0%B5%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D0%B8_%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC%D0%BE%D0%B2/%D0%A0%D0%B0%D1%81%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5_%D0%9B%D0%B5%D0%B2%D0%B5%D0%BD%D1%88%D1%82%D0%B5%D0%B9%D0%BD%D0%B0
# Levenshtein distance between a and b
def distance(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n, m)) space
        a, b = b, a
        n, m = m, n

    current_row = range(n + 1)  # Keep current and previous row, not entire matrix
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]
                
# punct_symbols = ['(', ')', ',', '/', '"', "'",':' ';' '_' ]
# punct_symbols = ['(', ')', ','] 
PUNCT_SYMBOLS = string.punctuation

# import requests
# import lxml.html
# res = requests.get('https://russky.info/ru/grammar/prepositions')
# html = lxml.html.fromstring(res.content)
# PREPOSITIONS = set([t.text for t in html.xpath("//span[@data-sound ]//span[@class='orig']/b")])
PREPOSITIONS = {'в',
 'да',
 'для',
 'до',
 'и',
 'из-за',
 'или',
 'как', 
 'так',
 'либо',
 'на',
 'не','только', 'но',
 'то',
 'о',
 'от',
 'перед',
 'по',
 'под',
 'с',
 'то', 'ли', 'к', 'при'}
# Не хватает сложных "кое-как", "как-нибудь" и т.д.

#https://pymorphy2.readthedocs.io/en/latest/user/grammemes.html
# NPRO	местоимение-существительное	он
# PRED	предикатив	некогда
# PREP	предлог	в
# CONJ	союз	и
# PRCL	частица	бы, же, лишь
# INTJ	междометие	ой
RESTRICT_TAGS={'NPRO','PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ'}

def _join(words, sort=True, lower=True):
    if sort:
        res = ' '.join(sorted(words)).strip()
    else:
        res = ' '.join(list(words)).strip()

    if lower:
        return res.lower()
    else:
        return res

def normalize_ma(sentence, out_of_vocab2synonym=None, 
                 sort=False, unique=False, punct_symbols=None, lower=True,
                 ma=None, restrict_tags=None, accept_tags=None,
                 min_word_len=0,
                 *args, **kwargs):
    if out_of_vocab2synonym is None:
        out_of_vocab2synonym = dict()

    if ma is None:
        ma = pymorphy2.MorphAnalyzer()
    if restrict_tags is None:
        restrict_tags = RESTRICT_TAGS
        
    if punct_symbols is None:
        punct_symbols=list()
        
    for s in punct_symbols:
        sentence = sentence.replace(s,' ')
    sentence = sentence.strip()
        
    words = set([w.strip() for w in sentence.split(' ')])
    words = set([w for w in words if len(w) > min_word_len])
    words_parsed = [ma.parse(w) for w in words]
    words_ = set()
    
    for word, parsed_word in zip(words, words_parsed):
        word_tags = parsed_word[0].tag
        
        w_l = word.lower()
        out_of_vocab2synonym[word] = w_l
        word = w_l
        
        if accept_tags is not None:
            if not any([t in word_tags for t in accept_tags]):
                continue
         
        if (any([t in word_tags
                 for t in restrict_tags
                ]
               )
           ):
            continue
            
        words_.add(word)
        if 'Geox' in parsed_word[0].tag :
            word_ = word[0].upper()+word[1:]
            for sep in ['-', '—',]:
                splitted =  word.split(sep)
                if len(splitted)>1:
                    word_ = sep.join([t[0].upper()+t[1:] if len(t)>1 else t
                                      for t in splitted])
            out_of_vocab2synonym[word] = word_
            
    words = set(words_)

    if not unique:
        words_ = [out_of_vocab2synonym[w.strip()] 
                  if w.strip() in out_of_vocab2synonym else w.strip()
                  for w in sentence.split(' ') 
                  if (w.strip() in words or 
                      (w.strip() in out_of_vocab2synonym and out_of_vocab2synonym[w.strip()] in words))
                 ]
        words = words_
        
    if out_of_vocab2synonym is not None:
        words_ = list()
        for w in words:
            if w in out_of_vocab2synonym:
                words_.append(out_of_vocab2synonym[w])
            else:
                words_.append(w)

        words = words_
    
    words = [w for w in words if len(w) > min_word_len]

    return _join(words, sort, lower)


def normalize_ma_lemmatize(sentence, out_of_vocab2synonym=None,
                           sort=False, unique=False, punct_symbols=None, lower=False,
                           ma=None, restrict_tags=None, accept_tags=None,
                           min_word_len=0,
                           *args, **kwargs):
    if out_of_vocab2synonym is None:
        out_of_vocab2synonym = dict()

    if ma is None:
        ma = pymorphy2.MorphAnalyzer()
    if restrict_tags is None:
        restrict_tags = RESTRICT_TAGS

    if punct_symbols is None:
        punct_symbols = list()

    for s in punct_symbols:
        sentence = sentence.replace(s, ' ')
    sentence = sentence.strip()

    words = set([w.strip() for w in sentence.split(' ')])
    words = set([w.strip() for w in words if len(w) > min_word_len])
    words_parsed = [ma.parse(w) for w in words]
    words_ = set()
    #     print (words)
    for word, parsed_word in zip(words, words_parsed):
        normal_form = parsed_word[0].normal_form
        if len(word) > 2:
            out_of_vocab2synonym[word] = normal_form
            word = normal_form
        else:
            w_l = word.lower()
            out_of_vocab2synonym[word] = w_l
            word = w_l

        word_tags = parsed_word[0].tag

        if accept_tags is not None:
            if not any([t in word_tags for t in accept_tags]):
                continue

        if (any([t in word_tags
                 for t in restrict_tags
                 ]
                )
        ):
            continue

        words_.add(word)
        if 'Geox' in parsed_word[0].tag:
            word_ = word[0].upper() + word[1:]
            for sep in ['-', '—', ]:
                splitted = word.split(sep)
                if len(splitted) > 1:
                    word_ = sep.join([t[0].upper() + t[1:] if len(t) > 1 else t
                                      for t in splitted])
            out_of_vocab2synonym[word] = word_

    words = set(words_)
    #     print (words)
    #
    #     print (out_of_vocab2synonym)

    if not unique:
        words_ = [out_of_vocab2synonym[w.strip()]
                  if w.strip() in out_of_vocab2synonym else w.strip()
                  for w in sentence.split(' ')
                  if (w.strip() in words or
                      (w.strip() in out_of_vocab2synonym and out_of_vocab2synonym[w.strip()] in words))
                  ]
        words = words_

    #     print (words)

    if out_of_vocab2synonym is not None:
        words_ = list()
        for w in words:
            if w in out_of_vocab2synonym:
                words_.append(out_of_vocab2synonym[w])
            else:
                words_.append(w)

        words = words_

    words = [w.replace('ё', 'е') for w in words]
    words = [w.replace('Ё', 'Е') for w in words]

    #     print (words)

    return _join(words, sort, lower)

####################################################################################
################### RELEVANCE RE-CALCULATION UTILS #################################
####################################################################################
def _get_hypernyms(synsetid, gold_synsetid2parents, account_gold=True, ruwordnet=None):
    if account_gold and synsetid in gold_synsetid2parents:
        return gold_synsetid2parents[synsetid]
    else:
        return ruwordnet.get_hypernyms_by_id(synsetid)

def get_top_hyperomyns_counter(w, k=10, p1=0.0, p2=1.0, p3=0.0, 
                                  account_gold=True,
                                  ruwordnet_matrix=None,
                                  gold_synsetid2parents=None,
                                  synsetstr2id=None,
                                  synsetstr2vector=None,
                                  model=None,
                                  ruwordnet=None
                              ):
    
    __get_word_vector = get_word_vector_func(model)
    
    tmp_res = [((synsetstr2id[s],
                 _get_hypernyms(synsetstr2id[s], gold_synsetid2parents, account_gold, ruwordnet)
                ),rate 
               )
               for rate, s in get_top(__get_word_vector(w),
                                      ruwordnet_matrix,
                                      words=list(synsetstr2vector.keys()),k=k)]

    counter_hyperonyms = collections.Counter()
    for (h1, p_hyperonyms), rate in tmp_res:
        if p1>=0.0:
            counter_hyperonyms[h1] += p1*rate

        for h2 in p_hyperonyms:
            counter_hyperonyms[h2] += p2*rate
            if p3 < 0.0:
                continue
            for h3 in _get_hypernyms(h2, gold_synsetid2parents, account_gold, ruwordnet):
                counter_hyperonyms[h3] += p3*rate

    return counter_hyperonyms

def get_top_hyperomyns_counter_v(v, k=10, p1=0.0, p2=1.0, p3=0.0, 
                                  account_gold=True,
                                  ruwordnet_matrix=None,
                                  gold_synsetid2parents=None,
                                  synsetstr2id=None,
                                  synsetstr2vector=None,
                                  ruwordnet=None
                              ):
    
    tmp_res = [((synsetstr2id[s],
                 _get_hypernyms(synsetstr2id[s], gold_synsetid2parents, account_gold, ruwordnet)
                ),rate 
               )
               for rate, s in get_top(v,
                                      ruwordnet_matrix,
                                      words=list(synsetstr2vector.keys()),k=k)]

    counter_hyperonyms = collections.Counter()
    for (h1, p_hyperonyms), rate in tmp_res:
        if p1>=0.0:
            counter_hyperonyms[h1] += p1*rate

        for h2 in p_hyperonyms:
            counter_hyperonyms[h2] += p2*rate
            if p3 < 0.0:
                continue
            for h3 in _get_hypernyms(h2, gold_synsetid2parents, account_gold, ruwordnet):
                counter_hyperonyms[h3] += p3*rate

    return counter_hyperonyms

####################################################################################
################### RUWORDNET MATRIX CREATION UTILS ################################
####################################################################################
def get_senses_by_synset_id(ruwordnet, synset_id):
    return [sense for sense in [ruwordnet.get_sense_by_id(sense_id) 
                  for sense_id in ruwordnet.synsetid2senseids[synset_id]]
           ]

def get_senses_texts_by_synset_id(ruwordnet, synset_id):
    return [(sense['name'], sense['lemma'], sense['main_word'])
            for sense in get_senses_by_synset_id(ruwordnet, synset_id)]

def get_word_vector_func(model):
    if isinstance(model, (fasttext.FastText._FastText)):
        __get_word_vector = model.get_word_vector
    elif isinstance(model, (gensim.models.fasttext.FastText)):
        __get_word_vector = model.wv.get_vector
    elif isinstance(model, (gensim.models.keyedvectors.FastTextKeyedVectors)):
        __get_word_vector = model.get_vector
    elif hasattr(model, 'get_word_vector'):
        __get_word_vector = model.get_word_vector
    elif hasattr(model, 'wv') and hasattr(model.wv, 'get_vector'):
        __get_word_vector = model.wv.get_vector
    elif hasattr(model, 'get_vector'):
        __get_word_vector = model.get_vector
    else:
        raise AttributeError (f"Can't get 'get_vector' or 'get_word_vector' method from {model}")

    return __get_word_vector


def get_sentence_vector_avg_words(model, sentence):
    __get_word_vector = get_word_vector_func(model)
    
    if isinstance(sentence, (str)):
        words = sentence.split()
    elif isinstance(sentence, (list)):
        words = sentence
        
    if len(words) < 1:
        print ("ERROR:" , words)

    vector_0 = __get_word_vector(words[0])
    sentence_matrix = np.zeros((len(words), vector_0.shape[0]),
                               vector_0.dtype)
    sentence_matrix[0] = vector_0
    for i,w in enumerate(words[1:]):
        sentence_matrix[i+1] = __get_word_vector(w)
    return np.mean(sentence_matrix,0)


def get_sentence_vector(model, sentence):
    if isinstance(model, (fasttext.FastText._FastText)):
        return model.get_sentence_vector(sentence)
    elif hasattr(model, 'get_sentence_vector'):
        return model.get_sentence_vector(sentence)
    else:
        return get_sentence_vector_avg_words(model, sentence)

    
def get_avg_sentences_vector(sentences, model):
    sentences = list(sentences)
    if sentences[0].strip() != '':
        vector_0 = get_sentence_vector(model, sentences[0].strip())
        sentence_matrix = np.zeros((len(sentences), vector_0.shape[0]),
                                   vector_0.dtype)
    else:
        raise Exception ("Empty sentence[0]! % " % (sentences))

    sentence_matrix[0] = vector_0
    for i,sentence in enumerate(sentences[1:]):
        if sentence.strip() != '':
            sentence_matrix[i+1] = get_sentence_vector(model, sentence)
        else:
            print(f"Exception: Empty sentences[{i}] == ''. Sentences: {sentences}")
    return np.mean(sentence_matrix,0)


# def get_synset_str_and_vector(ruwordnet, 
#                               synset_id, 
#                               sort=True, 
#                               unique=True,
#                               ruthes_name=True,
#                               definition=False,
#                               senses_names=True,
#                               senses_lemmas=True,
#                               senses_main_word=False,
#                               norm_function='lower',
#                               model=None,
#                               sep=' '):
    
#     synset = ruwordnet.synsets_list[ruwordnet.synsetid2synsetnum[synset_id]]
#     words = list()
#     if ruthes_name:
#         words.extend([w for w in list(synset['ruthes_name'].split(', '))])
#     if definition:
#         words.extend([w for w in list(synset['definition'].split(', '))])

#     for s_name, s_lemma, s_main_word in get_senses_texts_by_synset_id(ruwordnet, synset_id):
#         if senses_names:
#             words.extend(s_name.split(', '))
#         if senses_lemmas:
#             words.extend(s_lemma.split(', '))
#         if senses_main_word:
#             words.extend(s_main_word.split(', '))
    
#     if norm_function=='lower':
#         words = [w.strip().lower() 
#                  for w in words
#                 ]
#     else:
#         words = [norm_function(w, sort=sort, unique=unique).strip().lower()
#                  for w in words
#                  ]
#     words = [w for w in words if w != '']

#     if unique:
#         words = list(set(words))
    
#     if sort:
#         words = sorted(words)
    
#     return words, get_avg_sentences_vector(words, model)

# def get_synset_words(ruwordnet, 
#                    synset_id, 
#                    norm_params,
#                    make_sent_params,
#                    norm_function='lower'):
#     synset = ruwordnet.synsets_list[ruwordnet.synsetid2synsetnum[synset_id]]
#     words = list()
#     if 'ruthes_name' in make_sent_params and make_sent_params['ruthes_name']:
#         words.extend([w for w in list(synset['ruthes_name'].split(', '))])
#     if 'definition' in make_sent_params and make_sent_params['definition']:
#         words.extend([w for w in list(synset['definition'].split(', '))])

#     for s_name, s_lemma, s_main_word in get_senses_texts_by_synset_id(ruwordnet, synset_id):
#         if 'senses_names' in make_sent_params and make_sent_params['senses_names']:
#             words.extend(s_name.split(', '))
#         if 'senses_lemmas' in make_sent_params and make_sent_params['senses_lemmas']:
#             words.extend(s_lemma.split(', '))
#         if 'senses_main_word' in make_sent_params and make_sent_params['senses_main_word']:
#             words.extend(s_main_word.split(', '))
    
#     if norm_function=='lower':
#         words = [w.strip().lower() 
#                  for w in words
#                 ]
#     else:
#         words = [norm_function(w, **norm_params).strip()
#                  for w in words
#                 ]

#     return [w for w in words if w != '']
    

# def get_synset_str_and_vector(ruwordnet, 
#                               synset_id, 
#                               norm_params,
#                               make_sent_params,
#                               norm_function='lower',
#                               model=None,
#                               sep=' '):
    
#     words = get_synset_words(ruwordnet, 
#                              synset_id, 
#                              norm_params,
#                              make_sent_params,
#                              norm_function)

#     if 'unique' in norm_params and norm_params['unique']:
#         words = list(set(words))
#     if 'sort' in norm_params and norm_params['sort']:
#         words = sorted(words)
    
#     return words, get_avg_sentences_vector(words, model)

def get_synset_words(ruwordnet,
                     synset_id,
                     norm_params,
                     make_sent_params,
                     norm_function='lower'):
    def _cnv_w(w):
        if norm_function == 'lower':
            return w.lower()
        return norm_function(w, **norm_params).strip()

    synset = ruwordnet.synsets_list[ruwordnet.synsetid2synsetnum[synset_id]]
    words = list()
    if 'ruthes_name' in make_sent_params and make_sent_params['ruthes_name']:
        words.extend([_cnv_w(w.strip())
                      for w in list(synset['ruthes_name'].split(', '))])
    if 'definition' in make_sent_params and make_sent_params['definition']:
        words.extend([_cnv_w(w.strip())
                      for w in list(synset['definition'].split(', '))])

    for s_name, s_lemma, s_main_word in get_senses_texts_by_synset_id(ruwordnet, synset_id):
        if 'senses_names' in make_sent_params and make_sent_params['senses_names']:
            words.extend([_cnv_w(w.strip()) for w in s_name.split(', ')])
        if 'senses_lemmas' in make_sent_params and make_sent_params['senses_lemmas']:
            words.extend([_cnv_w(w.strip()) for w in s_lemma.split(', ')])
        if 'senses_main_word' in make_sent_params and make_sent_params['senses_main_word']:
            words.extend([_cnv_w(w.strip()) for w in s_main_word.split(', ')])

    return [w for w in words if w.strip() != '']


def get_synset_str_and_vector(ruwordnet,
                              synset_id,
                              norm_params,
                              make_sent_params,
                              norm_function='lower',
                              model=None,
                              sep=' '):
    words = get_synset_words(ruwordnet,
                             synset_id,
                             norm_params,
                             make_sent_params,
                             norm_function)

    if 'unique' in norm_params and norm_params['unique']:
        words = list(set(words))
    if 'sort' in norm_params and norm_params['sort']:
        words = sorted(words)

    return words, get_avg_sentences_vector(words, model)


def get_synset_words_lemma(ruwordnet,
                           synset_id,
                           norm_params,
                           make_sent_params,
                           norm_function='lower'):
    def _cnv_w(w):
        if norm_function == 'lower':
            return w.lower()
        return norm_function(w, **norm_params).strip()

    synset = ruwordnet.synsets_list[ruwordnet.synsetid2synsetnum[synset_id]]
    words = list()
    if 'ruthes_name' in make_sent_params and make_sent_params['ruthes_name']:
        words.extend([_cnv_w(w.strip())
                      for w in list(synset['ruthes_name'].split(', '))])
    if 'definition' in make_sent_params and make_sent_params['definition']:
        words.extend([_cnv_w(w.strip())
                      for w in list(synset['definition'].split(', '))])

    for s_name, s_lemma, s_main_word in get_senses_texts_by_synset_id(ruwordnet, synset_id):
        if 'senses_names' in make_sent_params and make_sent_params['senses_names']:
            words.extend([w.lower() for w in s_lemma.split(', ')])
        if 'senses_lemmas' in make_sent_params and make_sent_params['senses_lemmas']:
            words.extend([w.lower() for w in s_lemma.split(', ')])
        if 'senses_main_word' in make_sent_params and make_sent_params['senses_main_word']:
            #words.extend([w.lower() for w in s_lemma.split(', ')]) 
            # fixed 2020-03-23 (after article submitting)
            words.extend([_cnv_w(w.strip()) for w in s_main_word.split(', ')])

    return [w for w in words if w.strip() != '']


def get_synset_str_and_vector_lemma(ruwordnet,
                                    synset_id,
                                    norm_params,
                                    make_sent_params,
                                    norm_function='lower',
                                    model=None,
                                    sep=' '):
    words = get_synset_words_lemma(ruwordnet,
                                   synset_id,
                                   norm_params,
                                   make_sent_params,
                                   norm_function)

    if 'unique' in norm_params and norm_params['unique']:
        words = list(set(words))
    if 'sort' in norm_params and norm_params['sort']:
        words = sorted(words)

    return words, get_avg_sentences_vector(words, model)