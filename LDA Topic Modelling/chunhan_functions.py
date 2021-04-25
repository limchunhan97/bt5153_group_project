#!/usr/bin/env python

import re 
import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import langdetect
from langdetect import detect
import spacy

# remove characters
def remove_punct(text):
    return re.sub(r'[`\-=~!@#$%^&*()\_+—\[\]{};\'\\:"|<,./<>?]', '', text)

# stemming
def words_stemmer(words, type="PorterStemmer", lang="english", encoding="utf8"): 
    supported_stemmers = ["PorterStemmer","LancasterStemmer","SnowballStemmer"]
    words = nltk.word_tokenize(words)
    if type is False or type not in supported_stemmers:
        return words
    else:
        stem_words = []
        if type == "PorterStemmer":
            stemmer = PorterStemmer()
            for word in words:
                stem_words.append(stemmer.stem(word))
        if type == "LancasterStemmer":
            stemmer = LancasterStemmer()
            for word in words:
                stem_words.append(stemmer.stem(word))
        if type == "SnowballStemmer":
            stemmer = SnowballStemmer(lang)
            for word in words:
                stem_words.append(stemmer.stem(word))
        return " ".join(stem_words)

def find_pos(word):
    # Part of Speech constants
    # ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'

    pos = nltk.pos_tag(nltk.word_tokenize(word))[0][1]
    
    # Adjective tags - 'JJ', 'JJR', 'JJS'
    if pos.lower()[0] == 'j':
        return 'a'
    # Adverb tags - 'RB', 'RBR', 'RBS'
    elif pos.lower()[0] == 'r':
        return 'r'
    # Verb tags - 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
    elif pos.lower()[0] == 'v':
        return 'v'

    # Noun tags - 'NN', 'NNS', 'NNP', 'NNPS'
    else:
        return 'n'

# Function to apply lemmatization to a list of words
def words_lemmatizer(text, encoding="utf8"):
    words = nltk.word_tokenize(text)
    lemma_words = []
    wl = WordNetLemmatizer()
    for word in words:
        pos = find_pos(word)
        lemma_words.append(wl.lemmatize(word, pos))
    return " ".join(lemma_words)


# Drop independent numbers (not alphanumeric) in sentences
def remove_numbers(text):
    words = [i for i in text.split(" ") if not i.isnumeric()]
    return " ".join(words)

def remove_repeat_char(text):
    return re.sub(r"(\w)\1{2,}",r"\1", text)

# remove stop words
def remove_stopwords(text, lang='english'):
    words = nltk.wordpunct_tokenize(text)
    lang_stopwords = stopwords.words(lang)
    stopwords_removed = [w for w in words if w.lower() not in lang_stopwords]
    return " ".join(stopwords_removed)

def preprocess_punct(text):
    # translate Chinese punct to English versions
    E_pun = u',.!?[]()<>""\'\''
    C_pun = u'，。！？【】（）《》“”‘’'
    table= {ord(f):ord(t) for f,t in zip(C_pun,E_pun)}
    return text.translate(table)

def remove_standalone_alphabets(text):
    words = nltk.wordpunct_tokenize(text)
    # a, i and u don't need to remove now for sentence completeness
    chra = list('bcdefghjklmnopqrstvwxyz')
    chra_removed = [w for w in words if w.lower() not in chra]
    return " ".join(chra_removed)

def remove_html_tag(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# Filter for only nouns
def noun_only(text):
    pos_comment = [nltk.pos_tag(word_tokenize(sent)) for sent in sent_tokenize(text)]
    
    filtered = []
    
    for sentence in range(len(pos_comment)):
        
        pos_comment_sent = pos_comment[sentence]
        
        filtered_sent = [word[0] for word in pos_comment_sent if word[1] in ['NN'] or word[1] in ['NNP'] or word[1] in ['NNS'] or word[1] in ['NNPS']]
        
        filtered.append(filtered_sent)
    
    filtered = [word for sent in filtered for word in sent]
    
    filtered = ' '.join(filtered)
    
    return filtered

nlp = spacy.load("en_core_web_sm")

def lemma(text):
    """
    Lemmatize comments using spacy lemmatizer.
    :param comment: a comment
    :return: lemmatized comment
    """
    lemmatized = nlp(text)
    lemmatized_final = ' '.join([word.lemma_ for word in lemmatized if word.lemma_ != '\'s'])
    return lemmatized_final

def clean_text(text):

    text = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = remove_html_tag(text)

    text = remove_emoji(text)

    text = preprocess_punct(text)

    text = re.sub('(\\[(.*?)])', ' ', text)

    text = re.sub('(\\(.*?)\\)', '', text)

    text = re.sub('(\\{.*?)\\}', '', text)
    
    text = noun_only(text)
    
    text = lemma(text)

    text = text.lower()

    text = replace_contractions(text)

    text = remove_punct(text)

    text = remove_numbers(text)
    text = remove_stopwords(text)

    text = re.sub('([^A-Za-z0-9])', ' ', text)

    text = remove_standalone_alphabets(text)

    text = ' '.join(text.split())

    return text