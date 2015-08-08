import nltk
from nltk.collocations import BigramCollocationFinder
from toolz.itertoolz import get
from toolz.functoolz import partial


class BigramLabelFinder(object):
    def __init__(self, measure='pmi', pos=None):
        """
        measure: str
            the measurement method, 'pmi'or 'chi_sq'
        """
        self.bigram_measures = nltk.collocations.BigramAssocMeasures()
        assert measure in ('pmi', 'chi_sq')
        self._measure_method = measure

        self._pos = pos
    
    def find(self, docs, min_freq, top_n):
        """
        Parameter:
        ---------------

        docs: list of tokenized documents
            
        min_freq: int
            the minimal frequency to be considered

        top_n: int
            how many labels to return

        Return:
        ---------------
        list of tuple of str: the bigrams
        """
        # if apply pos constraints
        # check the pos properties
        if self._pos:
            assert isinstance(self._pos, list)
            for pair in self._pos:
                assert isinstance(pair, tuple) or isinstance(pair, list)
                assert len(pair) == 2  # because it's bigram

        score_func = getattr(self.bigram_measures,
                             self._measure_method)

        finder = BigramCollocationFinder.from_documents(docs)
        finder.apply_freq_filter(min_freq)

        if self._pos:
            valid_pos_tags = set([pair for pair in self._pos])
            valid_bigrams = []
            bigrams = map(partial(get, 0),  # get the bigram
                          finder.score_ngrams(score_func))
            cnt = 0
            for bigram in bigrams:
                if tuple(map(partial(get, 1), bigram)) in valid_pos_tags:
                    valid_bigrams.append(bigram)
                    cnt += 1
                if cnt == top_n:  # enough
                    break
            return valid_bigrams
        else:
            bigrams = finder.nbest(score_func,
                                   top_n)
            return bigrams
            