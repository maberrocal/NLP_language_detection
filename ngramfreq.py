# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:32:51 2017

@author: Miguel Ángel Berrocal
"""

import operator
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
import pandas as pd


class NgramProfileGenerator(object):
    def __init__(self):

        """Constructor"""

        pattern = "[a-zA-Z'`éèî]+"

        # pattern = r'''(?x)          # set flag to allow verbose regexps
        #    (?:[A-Z]\.)+        # abbreviatiosn, e.g. U.S.A.
        #  | \w+(?:-\w+)*        # words with optional internal hyphens
        #  | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
        # '''

        self._tokenizer = RegexpTokenizer(pattern)

    def _generate_ngrams_ocurrences(self, tokens):
        """
        Scan down each token, generating all possible N-grams, for N=2 to 5.
        Use positions that span the padding blanks, as well.

        @param tokens: List of tokens
        @type tokens: list

        @return: List of generated n-grams(non repeated)
        @rtype: dictionary
        """

        ngrams_ocurrences = {}

        for token in tokens:

            for x in range(3, 4):  # generate N-grams, for N=1 to 5

                mod_token = "_" + token + "_"
                xngrams = ngrams(mod_token, x)

                for xngram in xngrams:

                    ngram = ''.join(xngram)

                    if ngram not in ngrams_ocurrences:
                        ngrams_ocurrences.update({ngram: 1})
                    else:
                        ngram_occurrences = ngrams_ocurrences[ngram]
                        ngrams_ocurrences.update({ngram: ngram_occurrences + 1})

        return ngrams_ocurrences

    def generate_ngram_frequency_profile(self, raw_text):

        """
        @param raw_text: Text to create a Ngram frequency profile
        @type raw_text: str

        @return: Ngram frequency profile
        @rtype: tuple list
        """

        tokens = self._tokenizer.tokenize(raw_text)
        ngrams_unsorted_profile = self._generate_ngrams_ocurrences(tokens)

        # Ngramas generados en una frase
        length = len(ngrams_unsorted_profile.items())
        end = length if length < 350 else 350

        ngrams_sorted_profile = sorted(ngrams_unsorted_profile.items(), key=operator.itemgetter(1), reverse=True)[0:end]

        return ngrams_sorted_profile


#####################################
#   LANGUAGE CLASSIFICATION CLASS   #
#####################################

class LanguageClassifier(object):
    def __init__(self, language_profiles):

        """Constructor"""

        self.language_profiles = language_profiles

    def guess_language(self, file_path):

        """
        @param file_path: Path
        @type file_path: str

        @return: Guessed language
        @rtype: str
        """
        raw_text = open(file_path, "r", encoding="utf-8").read()
        guessed_language = self._set_distance(raw_text)

        return guessed_language


    def guess_language_by_row(self, file_path):

        """
        @param file_paht: Path
        @type file_path: str

        @return: Guessed language
        @rtype: str
        """

        #raw_text = open(file_path, "r", encoding="utf-8").read()
        #rows = raw_text.split("\n")
        entrada = pd.read_csv(file_path, sep='|', header=0, encoding='utf-8')
        #Inicializo la nueva columna de prediccion
        entrada['idioma_pred'] = ''
        dfallos = {}

        for i in pd.unique(entrada['idioma']):
            dfallos[i] = []

        for index, row in enumerate(entrada['frase']):
            guessed_language = self._set_distance(row)
            if guessed_language != entrada['idioma'][index]:
                print("Sentence: {}... -> row: {} -> Suggested language: {} -> Real language: {}".format(row[0:25], index,guessed_language,entrada['idioma'][index]))
                # Como la frase ha fallado, insertamos en el diccionario de fallos para ajustar el entrenamiento
                dfallos[entrada['idioma'][index]].append(row)
            entrada['idioma_pred'][index] = guessed_language

        return entrada, dfallos

    def _set_distance(self, text):

        """
        @param text: Text to calculate distances
        @type text: str

        @return: Nearest distance language
        @rtype: str
        """

        languages_ratios = {}

        ngram_generator = NgramProfileGenerator()
        test_profile = ngram_generator.generate_ngram_frequency_profile(text)
        test_ngrams_sorted = [ngram[0] for ngram in test_profile]

        for language, ngrams_statistics in self.language_profiles.items():
            ngrams_only = [item for item in ngrams_statistics]
            distance = self._compare_ngram_frequency_profiles(ngrams_only, test_ngrams_sorted)

            languages_ratios.update({language: distance})

        nearest_language = min(languages_ratios, key=languages_ratios.get)

        return nearest_language

    def _compare_ngram_frequency_profiles(self, category_profile, document_profile):

        """
        @param category_profile, document_profile: Ngram profiles
        @type category_profile, document_profile: list

        @return: Document distance
        @rtype: int
        """

        document_distance = 0

        # Ngrams not stored in language profile
        maximum_out_of_place_value = len(category_profile) + 1

        for ngram in document_profile:
            # pick up index position of document ngram
            document_index = document_profile.index(ngram)
            try:
                # check if analyzed ngram exists in pre-computed language profile
                category_profile_index = category_profile.index(ngram)
                distance = abs(category_profile_index - document_index)
            except ValueError:
                '''
                If an Ngram is not in the language profile it takes the maximum out-of-place value.
                '''
                category_profile_index = maximum_out_of_place_value
                distance = category_profile_index

            '''
            The sum of all of the out-of-place values for all Ngrams is the
            distance measure for the document from the language profile.
            '''
            document_distance += distance

        return document_distance
