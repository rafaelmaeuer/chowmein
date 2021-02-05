import os
import nltk
from nltk.corpus import stopwords
import itertools
import codecs
from toolz.functoolz import compose
import pickle as pickle

language = 'german'
nltk.download('punkt')
nltk.download('stopwords')
CURDIR = os.path.dirname(os.path.realpath(__file__))

def load_line_corpus(path, tokenize=True):
    docs = []
    with codecs.open(path, "r", "utf8") as f:
        for l in f:
            if tokenize:
                sents = nltk.sent_tokenize(l.strip().lower(), language=language)
                docs.append(list(itertools.chain(*map(lambda p: nltk.word_tokenize(p, language=language), sents))))
            else:
                docs.append(l.strip())
    return docs


def load_nips(years=None, raw=False):
    # load data
    if not years:
        years = range(2008, 2015)
    files = ['nips-{}.dat'.format(year)
             for year in years]

    docs = []
    for f in files:
        docs += load_line_corpus('{}/datasets/{}'.format(CURDIR, f),
                                 tokenize=(not raw))
        
    return docs                


def load_lemur_stopwords():
    ger_stops = stopwords.words(language)
    return ger_stops
    # with codecs.open(CURDIR + '/../datasets/lemur-stopwords.txt', 
    #                  'r', 'utf8') as f:
    #     return map(lambda s: s.strip(),
    #                f.readlines())
