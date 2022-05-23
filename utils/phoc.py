'''
based on the phoc implementation by ssudholt
'''
import logging
import numpy as np

def build_phoc_descriptor(words, phoc_unigrams, unigram_levels,
                          split_character=None, on_unknown_unigram='nothing',
                          phoc_type='phoc'):
    '''
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).

    Args:
        word (str): word to calculate descriptor for
        phoc_unigrams (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels to use in the PHOC
        split_character (str): special character to split the word strings into characters
        on_unknown_unigram (str): What to do if a unigram appearing in a word
            is not among the supplied phoc_unigrams. Possible: 'warn', 'error', 'nothing'
        phoc_type (str): the type of the PHOC to be build. The default is the
            binary PHOC (standard version from Almazan 2014).
            Possible: phoc, spoc
    Returns:
        the PHOC for the given word
    '''
    # prepare output matrix
    logger = logging.getLogger('PHOCGenerator')
    if on_unknown_unigram not in ['error', 'warn','nothing']:
        raise ValueError('I don\'t know the on_unknown_unigram parameter \'%s\'' % on_unknown_unigram)
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)

    phocs = np.zeros((len(words), phoc_size))
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    # iterate through all the words
    for word_index, word in enumerate(words):
        if split_character is not None:
            word = word.split(split_character)

        n = len(word) #pylint: disable=invalid-name
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            if char not in char_indices:
                if on_unknown_unigram == 'warn':
                    logger.warn('The unigram \'%s\' is unknown, skipping this character', char)
                    continue
                elif on_unknown_unigram == 'error':
                    logger.fatal('The unigram \'%s\' is unknown', char)
                    raise ValueError()
                else:
                    continue
            char_index = char_indices[char]
            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        if phoc_type == 'phoc':
                            phocs[word_index, feat_vec_index] = 1
                        elif phoc_type == 'spoc':
                            phocs[word_index, feat_vec_index] += 1
                        else:
                            raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)

    return phocs