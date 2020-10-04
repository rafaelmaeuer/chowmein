import nltk
from toolz.functoolz import partial
from nltk.stem.porter import PorterStemmer

nltk.download('averaged_perceptron_tagger')

# Necessary for Python3
def list_filter(function, iterable):
    """my_filter(function or None, iterable) --> filter object

    Return an iterator yielding those items of iterable for which function(item)
    is true. If function is None, return the items that are true."""
    if function is None:
        return (item for item in iterable if item)
    return list(item for item in iterable if function(item))

# Necessary for Python3
list_map = lambda func, *iterable: list(map(func, *iterable))

class CorpusBaseProcessor(object):
    """
    Class that processes a corpus
    """
    def transform(self, docs):
        """
        Parameter:
        -----------
        docs: list of (string|list of tokens)
            input corpus
        
        Return:
        ----------
        list of (string|list of tokens):
            transformed corpus
        """
        raise NotImplemented


class CorpusWordLengthFilter(CorpusBaseProcessor):
    def __init__(self, minlen=2, maxlen=35):
        self._min = minlen
        self._max = maxlen

    def transform(self, docs):
        """
        Parameters:
        ----------
        docs: list of list of str
            the tokenized corpus
        """
        assert isinstance(docs[0], list)
        valid_length = (lambda word:
                        len(word) >= self._min and
                        len(word) <= self._max)
        filter_tokens = partial(list_filter, valid_length)
        return list_map(filter_tokens, docs)

porter_stemmer = PorterStemmer()


class CorpusStemmer(CorpusBaseProcessor):
    def __init__(self, stem_func=porter_stemmer.stem):
        """
        Parameter:
        --------------
        stem_func: function that accepts one token and stem it
        """
        self._stem_func = stem_func

    def transform(self, docs):
        """
        Parameter:
        -------------
        docs: list of list of str
            the documents

        Return:
        -------------
        list of list of str: the stemmed corpus
        """
        assert isinstance(docs[0], list)
        stem_tokens = partial(list_map, self._stem_func)
        return list_map(stem_tokens, docs)

class CorpusPOSTagger(CorpusBaseProcessor):
    def __init__(self, pos_tag_func=nltk.pos_tag):
        """
        Parameter:
        --------------
        pos_tag_func: pos_tag function that accepts list of tokens
            and POS tag them
        """
        self._pos_tag_func = pos_tag_func

    def transform(self, docs):
        """
        Parameter:
        -------------
        docs: list of list of str
            the documents

        Return:
        -------------
        list of list of str: the tagged corpus
        """
        assert isinstance(docs[0], list)
        return list_map(self._pos_tag_func, docs)
