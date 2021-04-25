#!/usr/bin/env python

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import multiprocessing as mp

def sentiment_score_gen(text):

    sia = SentimentIntensityAnalyzer()
    sent_dict = sia.polarity_scores(text)

    return sent_dict

# To be in jupyter notebook

#%%time

#import multiprocessing as mp

#from ural_functions import sentiment_score_gen

#if __name__ == '__main__':
    
#    process = mp.Pool(processes = 4)
#    list_dict = process.map(sentiment_score_gen, [i for i in X_cleaned])

