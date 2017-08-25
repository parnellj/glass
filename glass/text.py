from __future__ import division

import csv
import math
from collections import Callable, Counter, defaultdict, OrderedDict
import itertools
import numpy as np

from nltk import word_tokenize, PerceptronTagger
from nltk.corpus import cmudict, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))

class Characteristic:
    def __init__(self):
        pass

    def __str__(self):
        s = []
        for k in sorted(vars(self).keys()):
            v = vars(self)[k]
            if isinstance(v, list) or isinstance(v, tuple):
                s.append(k + ': ' + (str(v[0:9]) if len(v) >= 10 else str(v)))
            elif isinstance(v, str):
                s.append(k + ': ' + v)
        return '\n'.join(s)

def uniqify(seq, idfun=lambda x: x):
    """
    https://www.peterbe.com/plog/uniqifiers-benchmark, f5
    :param seq:
    :type seq:
    :param idfun:
    :type idfun:
    :return:
    :rtype:
    """
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

def formant_dict_loader(FORMANTS):
    dists = {}
    for i in FORMANTS.keys():
        dists[i] = {j: np.linalg.norm(np.array(FORMANTS[i][1:3] -
                                               np.array(FORMANTS[j][1:3])))
                    for j in FORMANTS.keys()}
    mean_dists = {}
    std_dists = {}
    for k, v in dists.iteritems():
        mean_dists[k] = np.mean(v.values())
        std_dists[k] = np.std(v.values())
    return dict(raw=dists, mean=mean_dists, std=std_dists)

def formant_tbl_loader(FORMANTS):
    """
    Calculate the Euclidean distance between all formant pairs with coordinates of f[1] to f[ord]
    :param o: Which CN.FORMANTS to include in the distance calculation
    :return: {(u'vowel_1', u'vowel_2'):int(distance)}
    """
    new = {}
    for i, j in itertools.product(FORMANTS.items(), repeat=2):
        new[(i[0], j[0])] = np.linalg.norm(np.array(i[1][1:3]) - np.array(j[1][1:3]))
    return new

class Constants:
    def __init__(self):
        pass

    EOFBRK = 'EOFBRK'
    STZBRK = 'STZBRK'
    LINBRK = 'LINBRK'
    TABBRK = 'TABBRK'
    WHITESPACE = [EOFBRK, STZBRK, LINBRK, TABBRK, '\n', '\t', '\r', '\f']
    FORMANTS = {u'AA': [215, 936, 1551, 2815, 4299], u'AE': [215, 669, 2349, 2972, 4290],
            u'AH': [218, 753, 1426, 2933, 4092], u'AO': [210, 781, 1136, 2824, 3923],
            u'AW': [222.5, 727.5, 1388, 2821, 4175.5],
            u'AY': [219.5, 709.5, 1958, 2934, 4316.5],
            u'EH': [214, 731, 2058, 2979, 4294], u'ER': [217, 523, 1588, 1929, 3914],
            u'EY': [219, 536, 2530, 3047, 4319], u'IH': [224, 483, 2365, 3053, 4334],
            u'IY': [227, 437, 2761, 3372, 4352], u'OW': [217, 555, 1035, 2828, 3927],
            u'OY': [217, 632, 1750.5, 2938.5, 4128.5],
            u'UH': [230, 519, 1225, 2827, 4052], u'UW': [235, 459, 1105, 2735, 4115]}
    FULL = ['A', 'B', 'A', 'B']
    PART = ['X', 'A', 'X', 'A']
    NONE = ['X', 'X', 'X', 'X']
    KNOWN_METERS = {(8, 6, 8, 6): ('Common', (FULL, FULL)),
                    (6, 6, 8, 6): ('Short', (FULL, PART)),
                    (8, 8, 8, 8): ('Long', (FULL, PART)),
                    (6, 6, 6, 6): ('Half', (FULL, PART))}
    NO_MATCH = ('No Match', (NONE, NONE))
    SAMPLE = ['bray', 'brain', 'grow', 'grown', 'tame', 'ended', 'ending', 'end', 'whatever',
              'whoever', 'bray', 'time', 'rhyme', 'thyme', 'flow', 'floe', 'forgiven',
              'hidden']
    SAMPLE_PHONEMES = [[u'B', u'R', u'EY1'], [u'B', u'R', u'EY1', u'N'],
                       [u'G', u'R', u'OW1'], [u'G', u'R', u'OW1', u'N'],
                       [u'T', u'EY1', u'M'], [u'EH1', u'N', u'D', u'AH0', u'D'],
                       [u'EH1', u'N', u'D', u'IH0', u'NG'], [u'EH1', u'N', u'D'],
                       [u'W', u'AH2', u'T', u'EH1', u'V', u'ER0'],
                       [u'HH', u'UW0', u'EH1', u'V', u'ER0'], [u'B', u'R', u'EY1'],
                       [u'T', u'AY1', u'M'], [u'R', u'AY1', u'M'], [u'TH', u'AY1', u'M'],
                       [u'F', u'L', u'OW1'], [u'F', u'L', u'OW1']]
    SAMPLE_PHONEMES_FLAT = [u'B', u'R', u'EY1', u'B', u'R', u'EY1', u'N', u'G', u'R',
                            u'OW1', u'G', u'R', u'OW1', u'N', u'T', u'EY1', u'M',
                            u'EH1', u'N', u'D', u'AH0', u'D', u'EH1', u'N', u'D',
                            u'IH0', u'NG', u'EH1', u'N', u'D', u'W', u'AH2', u'T',
                            u'EH1', u'V', u'ER0', u'HH', u'UW0', u'EH1', u'V', u'ER0',
                            u'B', u'R', u'EY1', u'T', u'AY1', u'M', u'R', u'AY1',
                            u'M', u'TH', u'AY1', u'M', u'F', u'L', u'OW1', u'F', u'L',
                            u'OW1']
    SAMPLE_LINES = [['bray', 'brain', 'grow', 'grown', 'tame'],
                    ['ended', 'ending', 'end', 'whatever', 'whoever', 'bray'],
                    ['time', 'rhyme', 'thyme', 'flow', 'floe']]

    FORMANT_DICT = formant_dict_loader(FORMANTS)
    FORMANT_TBL = formant_tbl_loader(FORMANTS)




class Tokens:
    def __init__(self, source):
        self.tokens = []
        if isinstance(source, str):
            self.initial_format(source)
        elif isinstance(source, list) and isinstance(source[0], str):
            self.tokens = [t for t in enumerate(source)]
        elif isinstance(source, list) and isinstance(source[0], tuple):
            self.tokens = source
        self.packet_size = len(self.tokens[0][1])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.tokens[index.start:index.stop:index.step]
        else:
            return self.tokens[:]

    def filter(self, types):
        filtered = self.tokens[:]
        for c in types:
            if c == 'no ws':
                filtered = [t for t in filtered if t[1] not in Constants.WHITESPACE]
            if c == 'lower':
                filtered = [(t[0], t[1].lower()) for t in filtered]
            if c == 'alnum':
                filtered = [t for t in filtered if t[1].isalnum()]
        return Tokens(filtered)

    def paste_over(self, new):
        for t in new:
            self.tokens[t[0]] = (t[0], t[1])
        return self

    def initial_format(self, string):
        """
        Provide an initial reformat and numbering of the imported text.  Notes:
            1.  Whitespace characters are replaced with constant strings.  This
                facilitates later slicing and also creates a linear, alphanumeric text.
            2.  Every glyph (word or punctuation) is enumerated.
        :return: A list of tuples in the format [(number, glyph), ...]
        :rtype: list
        """
        text = string
        text = text.replace('\n\n', pad(Constants.STZBRK, 1))
        text = text.replace('\r\n', pad(Constants.LINBRK, 1))
        text = text.replace('\n', pad(Constants.LINBRK, 1))
        text = text.replace('\t', pad(Constants.TABBRK, 1))
        text = word_tokenize(text)
        self.tokens = [w for w in enumerate(text)]

    def change_packet_size(self, n):
        num, tok = zip(*self.tokens)
        one_gram = [t[0] for t in tok] + [t for t in tok[-1][1:]]
        new_packets = [w for w in window(one_gram, n)]
        self.tokens = zip(num, new_packets)

    def get_tokens(self):
        return [t[1] for t in self.tokens]

    def get_positions(self):
        return [t[0] for t in self.tokens]

    def get_freqs(self):
        return Counter([t[1] for t in self.tokens])

def pad(t, n):
    return (' ' * n) + t + (' ' * n)

def window(seq, n=1):
    """
    Return a list of sliding window views into a sequence 'seq' of window size 'n'
    :param seq: A list of elements
    :type seq: list
    :param n: The width of the sliding window
    :type n: int
    :return: A list of windowed (overlapping) elements
    :rtype: list
    """
    i = 0
    end = len(seq) - n
    while i <= end:
        yield tuple(seq[i:i + n])
        i += 1

def flatten_list(lol):
    return [i for sublist in lol for i in sublist]

class Morphology(Characteristic):
    """
    Accounts for the morphological/lexicographical characteristics of the source text:
        Word-for-word:  Parses and stores word stems, lemmas, and punctuation.
        Entire text:    NA (morphology pertains only to individual words).
    """
    STEMMER = SnowballStemmer('english')
    LEMMATIZER = WordNetLemmatizer()

    def __init__(self, tokens):
        Characteristic.__init__(self)
        self.tokens = tokens
        self.stems = Morphology.stem(tokens.tokens)
        self.lemmas = Morphology.lemmatize(tokens.tokens)

    @classmethod
    def stem(cls, tokens):
        return [(t[0], cls.STEMMER.stem(t[1])) for t in tokens]

    @classmethod
    def lemmatize(cls, tokens):
        return [(t[0], cls.LEMMATIZER.lemmatize(t[1])) for t in tokens]



class Phonology(Characteristic):
    """
    Accounts for the phonetic characteristics of the source text. Notes:
        Word-for-word:  Parses and stores phonemes, stresses, vowels, consonants,
                        and final syllables.
        Entire text:    Detects and stores groups of rhyming words for different types
                        of rhyme (e.g., perfect, slant).  From this data the text's
                        rhyme scheme can also be derived.
    """

    def __init__(self, tokens):
        Characteristic.__init__(self)
        self.tokens = tokens
        self.phonemes = self.phonematize(self.tokens.tokens)

        self.scansion = self.scan(self.phonemes.tokens)
        self.vowels = self.filter_vowels(self.phonemes.tokens)
        self.consonants = self.filter_consonants(self.phonemes.tokens)
        self.final_syllables = self.filter_final_syllables(self.phonemes.tokens)

    @classmethod
    def phonematize(cls, tokens):
        return Tokens(zip([t[0] for t in tokens],
                          PhoneticDictionary.lookup([t[1] for t in tokens])))

    @classmethod
    def scan(cls, phonemes):
        return [(t[0], [ph[-1] for ph in t[1] if ph[-1].isdigit()]) for t in phonemes]

    @classmethod
    def strip_stresses(cls, phonemes):
        """
        Remove the stress digit from vowel phonemes in tokens
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :return: A list of unstressed phonemes in each token in tokens
        """
        return [[ph[:-1] if ph[-1].isdigit() else ph for ph in t] for t in phonemes]

    @classmethod
    def filter_vowels(cls, phonemes):
        """
        Return a list of vowels in tokens
        :param phonemes: A by-word, enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :return: A list of vowel phonemes in each token in tokens
        """
        return [(t[0], [ph for ph in t[1] if ph[-1].isdigit()]) for t in phonemes]

    @classmethod
    def filter_consonants(cls, phonemes):
        """
        Return a list of consonants in tokens
        :param phonemes: A by-word, enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :return: A list of consonant phonemes in each token in tokens
        """
        return [(t[0], [ph for ph in t[1] if not ph[-1].isdigit()]) for t in phonemes]

    @classmethod
    def filter_final_syllables(cls, phonemes, drop_left_consonant=True):
        """
        Return the last syllable from each token in phonemes.  Drop the consonant to the
        left of the last vowel by default.
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :param drop_left_consonant: Drop the consonant left (prior to) the vowel if True
        :return: A list of final syllable phonemes in each token in tokens
        """
        return []

    @classmethod
    def measure_phoneme_frequency(cls, phonemes):
        """
        Generate absolute frequencies of all phonemes in the passed list.
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A Counter of matching phonemes
        :rtype: Counter
        """
        phoneme_tokens_flat = flatten_list([t[1] for t in phonemes])
        phonemes = [p[:-1] if p[-1].isdigit() else p for p in phoneme_tokens_flat]
        return Counter(phonemes)


    @classmethod
    def find_meter(cls):
        meter = []
        return meter

    @classmethod
    def find_rhymes(cls):
        pass

class Formants:
    def __init__(self):
        pass

    @classmethod
    def closeness_a_b(cls, a, b):
        dist = Constants.FORMANT_DICT['raw'][a][b]
        mean = Constants.FORMANT_DICT['mean'][a]
        std = a.FORMANT_DICT['std'][a]
        direction = 'above' if dist > mean else 'below'
        z = (dist - mean) / std
        # print a + ' to ' + b + ' raw dist: ' + str(dist)
        # print str(z) + ' standard deviations'
        # print direction + ' the ' + a + ' to ' + b + ' mean of ' + str(mean)
        return z

    @classmethod
    def closeness_pop(cls, a, b):
        dist = Constants.FORMANT_DICT['raw'][a][b]
        pop_mean = np.mean([sub.values() for sub in Constants.FORMANT_DICT['raw'].values()])
        pop_std = np.std([sub.values() for sub in Constants.FORMANT_DICT['raw'].values()])
        pop_dir = 'above' if dist > pop_mean else 'below'
        z = (dist - pop_mean) / pop_std
        # print a + ' to ' + b + ' raw dist: ' + str(dist)
        # print str((dist - pop_mean) / pop_std) + ' standard deviations'
        # print pop_dir + ' the population mean of ' + str(pop_mean)
        return z

    @classmethod
    def closeness_shape(cls, a, b):
        return


class PhoneticDictionary:
    CMUDICT = cmudict.dict()
    def __init__(self):
        pass

    @classmethod
    def append_words(cls, word_file):
        """
        Append entries at filename to the CMU Pronunciation Dictionary (cmudict)
        file should be in format:
        WORD    PH PH PH PH PH
        WORD    PH PH PH PH PH
        :param word_file: The full directory of the file
        :return: The updated cmudict
        """
        new_words = {}
        vowel_keys = Constants.FORMANTS.keys()
        with open(word_file, 'r') as f:
            for line in f.readlines():
                sep = line.index('\t')
                wd = line[:sep].lower()
                ph = line[sep + 1:].replace('\n', '').split(' ')
                ph = [[unicode(k) + u'1' if k in vowel_keys else unicode(k)
                       for k in ph]]
                new_words[wd] = ph
        cls.CMUDICT.update(new_words)

    @classmethod
    def get_misses(cls, tokens, write=False):
        """
        Compile words missing from the CMU Pronunciation Dictionary (cmudict).
        :param tokens: A list of tokens to be looked up in cmudict
        :param write: Whether or not to write the missing tokens to a file
        :return: A list of tokens not found in cmudict
        """
        fails = [t for t in tokens if not cls.CMUDICT.get(t, False)]
        fails = sorted(list(set(fails)))

        if write:
            with open('./outputs/cmu_missing.txt', 'w')as f:
                for w in fails: f.write(w + '\n')

        return fails

    @classmethod
    def lookup(cls, tokens):
        return [cls.CMUDICT.get(t, [[u'NOC', u'NOV0']])[0] for t in tokens]

class Rhymes:
    formants = Formants()

    def __init__(self, words, phonemes):
        self.words = words
        self.phonemes = phonemes

    @classmethod
    def rhyme_matches(cls, tokens):
        """
        Determine matching sets from a list of filtered tokens.  Only used for 'linear'
        rhyme schemes (e.g., 'identical', 'assonant')
        :param tokens: tokens filtered according to the specific rhyme scheme
        :return: dict
        """
        matches = DefaultOrderedDict(list)
        for a, b in itertools.combinations(tokens, 2):
            if a == b:
                matches[a].append(b)
        return matches

    @classmethod
    def set_scheme(cls, words, matches):
        """
        Create a rhyme scheme from the calculated word groups (see rhyme_matches()).
        :param words: the actual words being tested for rhymes
        :param matches: An (ordered) dict of matching groups
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        """
        scheme = ['X'] * len(words)
        letter = 'A'
        for k, v in matches.iteritems():
            indices = [i for i, w in enumerate(words) if w in v]
            for i in indices: scheme[i] = letter
            letter = chr(ord(letter) + 1)
        return scheme

    @classmethod
    def perfect(cls, phonemes):
        """
        Group words with identical final syllables.
        E.g.: "time", "rhyme"
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        last_syll = Phonology.filter_final_syllables(phonemes)
        last_syll = [tuple(a) for a in Phonology.strip_stresses(last_syll)]
        matches = cls.rhyme_matches(last_syll)
        scheme = cls.set_scheme(last_syll, matches)
        return scheme

    @classmethod
    def eye(cls, words, min_match=4):
        """
        Group words with the same 4 (or more) final characters.
        E.g.: "fre[ight]", "kn[ight]"
        :param words: A word-enumerated list of words: [(1, [word]),...]
        :type words: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        wds_r = [w[-min_match:] for w in words]
        matches = cls.rhyme_matches(wds_r)
        scheme = cls.set_scheme(wds_r, matches)
        return scheme

    @classmethod
    def identical(cls, words):
        """
        Group words that are spelled exactly the same.
        E.g.: "rhyme", "rhyme"
        :param words: A word-enumerated list of words: [(1, [word]),...]
        :type words: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        matches = cls.rhyme_matches(words)
        scheme = cls.set_scheme(words, matches)
        return scheme

    @classmethod
    def rich(cls, phonemes):
        """
        Group words that are pronounced exactly the same.  A superset of identical rhymes.
        E.g.: "flow", "floe"
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        no_stresses = [tuple(a) for a in Phonology.strip_stresses(phonemes)]
        matches = cls.rhyme_matches(no_stresses)
        scheme = cls.set_scheme(no_stresses, matches)
        return scheme

    @classmethod
    def assonant(cls, phonemes):
        """
        Group words that share an ultimate vowel phoneme.
        E.g.: "winn[o]w", "sl[o]pe"
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        final_vowels = [[w[-1]] for w in Phonology.filter_vowels(phonemes)]
        final_vowels = Phonology.strip_stresses(final_vowels)
        final_vowels = [tuple(v) for v in final_vowels]

        matches = cls.rhyme_matches(final_vowels)
        scheme = cls.set_scheme(final_vowels, matches)
        return scheme

    @classmethod
    def consonant(cls, phonemes):
        """
        Group words that share the same consonant phonemes.
        E.g.: "stump", "stamp"
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        consonants = Phonology.filter_consonants(phonemes)
        consonants = [tuple(v) for v in consonants]
        matches = cls.rhyme_matches(consonants)
        scheme = cls.set_scheme(consonants, matches)
        return scheme

    @staticmethod
    def trypop(vow, target):
        t = list(target)
        try:
            t.remove(vow)
        except ValueError:
            return -1
        else:
            return len(t)

    @classmethod
    def augmented(cls, phonemes):
        """
        Group words which differ only in the addition of a final consonantal phoneme
        E.g.: "bray" -> "brai[n]"
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        sylls = Phonology.filter_final_syllables(phonemes)
        terminal_vows = uniqify([s[-1][:-1] for s in sylls if s[-1][-1].isdigit()])
        sylls = Phonology.strip_stresses(sylls)

        matches = []
        letter = 'A'

        for vowel in terminal_vows:
            match = [cls.trypop(vowel, s) for s in sylls]
            if max(match):
                matches.append([letter + '__aug' if m >= 1 else
                                letter + '_base' if m == 0 else
                                'X' for m in match])
                letter = chr(ord(letter) + 1)

        if len(matches) == 0: matches.append(['X'] * len(phonemes))
        return list(zip(*matches))

    @classmethod
    def diminished(cls, phonemes):
        """
        Group words which differ only in the subtraction of a final consonantal phoneme
        E.g.: "brain" -> "bray[]"
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        sylls = Phonology.filter_final_syllables(phonemes)
        terminal_vows = uniqify([s[-1][:-1] for s in sylls if s[-1][-1].isdigit()])
        sylls = Phonology.strip_stresses(sylls)

        matches = []
        letter = 'A'

        for vowel in terminal_vows:
            match = [cls.trypop(vowel, s) for s in sylls]
            if max(match):
                matches.append([letter + '_base' if m >= 1 else
                                letter + '__dim' if m == 0 else
                                'X' for m in match])
                letter = chr(ord(letter) + 1)

        if len(matches) == 0: matches.append(['X'] * len(phonemes))
        return zip(*matches)

    @classmethod
    def unstressed(cls, phonemes):
        """
        Group words which rhyme perfectly with one another and whose vowel is
        unstressed (0 stress value in cmudict).
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        last_syll = [tuple(s) for s in Phonology.filter_final_syllables(phonemes)]
        matches = cls.rhyme_matches(last_syll)

        filtered_matches = OrderedDict()
        for k, v in matches.iteritems():
            try:
                if int(k[0][-1]) == 0: filtered_matches[k] = v
            except ValueError:
                continue
        scheme = cls.set_scheme(last_syll, filtered_matches)
        return scheme

    @classmethod
    def trailing(cls, phonemes):
        """

        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        scheme = ['X'] * len(phonemes)
        return scheme

    @classmethod
    def imputed(cls, stanzas):
        """
        Common:     Short:      Long:       Half:
        8 A         6 A A       8 A A       6 A A
        6 B         6 B B       8 B B       6 B B
        8 A         8 A X       8 A X       6 A X
        6 B         6 B B       8 B B       6 B B
        :param stanzas:
        :type stanzas:
        :return:
        :rtype:
        """

        stz_lens = [len(stz) for stz in stanzas]
        if not all([sl == 4 for sl in stz_lens]):
            return ('No Match', (['X'] * sum(stz_lens), ['X'] * sum(stz_lens)))

        line_lens = [[len(line) for line in stz] for stz in stanzas]
        nth_line_lens = zip(*line_lens)

        means = [sum(n) / len(n) for n in nth_line_lens]
        closest = [6 if abs(m - 6) <= abs(m - 8) else 8 for m in means]
        try:
            match = Constants.KNOWN_METERS[tuple(closest)]
        except KeyError:
            match = Constants.NO_MATCH

        letter = 'A'
        new_schemes = {0:[],
                       1:[]}

        for i in range(0, len(stanzas)):
            for j, sch in enumerate(match[1]):
                step = len(set([c for c in sch if c != 'X']))
                this_scheme = [chr(ord(c) + (i * step)) if c != 'X' else 'X' for c in sch]
                new_schemes[j].extend(this_scheme)

        return match[0], new_schemes.values()

    @classmethod
    def slant(cls, phonemes):
        """
        Group words with similar pronunciation. This is a highly complex type of rhyme that
        Dickinson used extensively.  The current function includes only the following two
        pairwise evaluations at present:
            1. Cartesian distance of ultimate vowel/diphthong phonemes < ????
            2. First and last letters of word the same
        :param phonemes: A word-enumerated list of phonemes: [(1, [ph_a, ph_b]),...]
        :type phonemes: list
        :return: A rhyme scheme in the format ['A', 'B', 'A', 'B', 'X',  ... ]
        :rtype: list
        """
        matches = DefaultOrderedDict(list)
        all_vowels = [[w[-1]] for w in Phonology.filter_vowels(phonemes)]
        all_vowels = [a[0] for a in Phonology.strip_stresses(all_vowels)]
        final_vowels = uniqify(all_vowels)

        letter = 'A'
        matches = []
        for v_a in final_vowels:
            degree = [Phonology.Formants.closeness_a_b(v_a, v_b) < -0.5 and v_a != v_b
                      for v_b in all_vowels]
            if any(degree):
                matches.append([letter + '_base' if all_vowels[i] == v_a else
                                letter + '_near' if m else
                                'X' for i, m in enumerate(degree)])
                letter = chr(ord(letter) + 1)

        if len(matches) == 0:
            return [tuple('X',)] * len(phonemes)
        else:
            return [tuple(m for m in z if m != 'X') for z in zip(*matches)]
# /class Rhymes


class Syntax(Characteristic):
    """
    Accounts for the syntactic aspects of the source text.
        Word-for-word:  Parses and stores part-of-speech (POS) tags.
        Entire text:    Enumerates all configurations of clause and sentence found in
                        the text.
    """
    POS_TAG = PerceptronTagger()

    def __init__(self, tokens):
        Characteristic.__init__(self)

    def pos_tag(self):
        pass

    def find_clauses(self):
        pass

    def find_sentences(self):
        pass

class Semantics(Characteristic):
    def __init__(self):
        Characteristic.__init__(self)

    @classmethod
    def wordnet_tag(cls):
        pass

class Text:
    """
    Represents several aspects of a discrete text:
        Metadata:           Raw and tokenized text, author, date, notes, etc.
        Characteristics:    The morphological, phonetic, syntactic, and semantic
                            attributes of the text (see class definitions).
        Statistical #s:     TBD
    """
    def __init__(self, text=None, text_type=None, author=None, date=None,
                 notes=None, number=None, biblio=None):

        # Metadata
        self.text = text
        self.tokens = Tokens(self.text[0])
        self.text_type = text_type
        self.author = author
        self.date = date
        self.notes = notes
        self.number = number
        self.biblio = biblio

        # Characteristics
        self.morph = Morphology(self.tokens.filter(['no ws', 'lower']))
        self.phone = Phonology(self.tokens.filter(['alnum', 'lower']))
        # self.syntax = Syntax()
        # self.semantics = Semantics()

        # Statistical Quantities


class Database:
    def __init__(self, texts, import_method, input_path):
        self.texts = []
        if import_method == 'csv':
            self.import_from_csv(input_path)
        elif import_method == 'dir':
            self.import_from_dir(input_path)
        if import_method == 'pickle':
            self.import_from_pickle(input_path)
        elif import_method == 'ris':
            self.import_from_ris(input_path)

    def import_from_csv(self, input_csv_path):
        with open(input_csv_path) as f:
            reader = csv.reader(f)
            heads = reader.next()
            content = []
            for row in reader:
                entry = defaultdict(list)
                for i, c in enumerate(row):
                    entry[heads[i]].append(c)
                content.append(entry)
        self.texts = [Text(**c) for c in content]

    def import_from_dir(self, input_dir, filename_re=None):
        self.texts = []

    def import_from_pickle(self, input_pickle_path):
        self.texts = []

    def import_from_ris(self, input_ris_path):
        self.texts = []

    def linearize(self):
        return [item for t in self.texts for item in t.morph.tokens.get_tokens()]

    def frequency(self, relative=False, sequence=None, gram=1):
        linear = [t for t in window(self.linearize(), gram)]
        length = len(linear)
        freq_abs = Counter(linear)
        if relative:
            return {k:(v/length) for k, v in Counter(linear).iteritems()}
        return freq_abs

    @classmethod
    def collocates(cls, word_counts, ngram_counts):
        ngram_pop_size = sum(ngram_counts.values())
        word_pop_size = sum(word_counts.values())

        word_freqs = {k:(v/word_pop_size) for k, v in word_counts.iteritems()}
        t_scores = dict()
        for ngram, count in ngram_counts.iteritems():
            #x-bar
            observed_freq = count / ngram_pop_size
            #mu
            null_freq = 1.0
            for word in ngram:
                null_freq *= word_freqs[(word,)]
            t_score = (observed_freq - null_freq) \
                      / math.sqrt(observed_freq / ngram_pop_size)
            t_scores[ngram] = t_score
        return t_scores

if __name__ == '__main__':
    d = Database(os.path.join('.', 'inputs', '2016-07-11_19h47m_db.csv'))
    m = d.texts[0].morph
    p = d.texts[0].phone
    # linear = d.linearize()
    # length = len(linear)
    # word_counts = d.frequency(gram=1)
    # bigram_counts = d.frequency(gram=2)
    # sw = stopwords.words('english')
    # collocates = d.collocates(word_counts, bigram_counts).items()
    # for k, v in collocates:
    #     if sum([True for w in k if w in sw]) == 0 and v >= 2.576:
    #         print str((k, bigram_counts[k], v))
    print 'main done'


class TextGenerator:
    def __init__(self):
        pass

print 'intiialization done'
