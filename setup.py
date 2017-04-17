# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:32:51 2017

@author: Miguel Ángel Berrocal
"""

import os
import ngramfreq
import pandas as pd
from pandas_ml import ConfusionMatrix

LANGDATA_FOLDER = 'corpus/txt'
LANGDATA_PROFILE = 'lang_profile'
TEST_PATH = "test/prueba.csv"

def get_language_profiles():

    dic_language_profiles = {}
    ngram_language_profiles = {}

    if "language_profile.csv" in os.listdir(LANGDATA_PROFILE):

        print("Language profile has been founded.")

        df = pd.read_csv(LANGDATA_PROFILE + "/" + "language_profile.csv")
        df_util = pd.DataFrame(columns=df.keys())
        df_a_guardar = pd.DataFrame(columns=df.keys())

        for i in df.keys():
            n_gram = df[i].str.extract('(\w{0,}[a-z]\w{0,})', expand=True)[0]  # Saca los n - gramas
            freq = df[i].str.extract('(\w{0,}[0-9]\w{0,})', expand=True)[0] # Saca las frecuencias
            df_a_guardar[i] = [[a, b] for a, b in zip(n_gram, freq)]
            df_util[i] = n_gram


        dic_language_profiles = df_a_guardar.to_dict(orient="list")
        ngram_language_profiles = df_util.to_dict(orient="list")

        print("Included languages files: {}".format(ngram_language_profiles.keys()))

        corpus_not_included = [name for name in os.listdir(LANGDATA_FOLDER) if
                            os.path.isfile(os.path.join(LANGDATA_FOLDER, name)) and name.endswith(".txt")
                            and name.split(".txt")[0] not in ngram_language_profiles.keys()]

        if corpus_not_included != []:
            print("New languages files have been detected: {}" .format(corpus_not_included))

            for language in corpus_not_included:

                language_name = language.split(".txt")[0]
                ngram_profile, ngram_freq = _ngram_profile(language)
                ngram_language_profiles[language_name] = ngram_profile
                dic_language_profiles[language_name] = [[a, b] for a, b in zip(ngram_profile, ngram_freq)]
                df = pd.DataFrame.from_dict(dic_language_profiles)
                df.to_csv(LANGDATA_PROFILE + "/" + "language_profile.csv", index=False)

            print("Language profile has been updated with: {}".format(corpus_not_included))
            print("Language profile contains :{}" .format(dic_language_profiles.keys()))

        else:
            print("Language profile is currently updated")

    else:

        print("Language profile has't been founded.")

        language_corpus = [name for name in os.listdir(LANGDATA_FOLDER)
                           if os.path.isfile(os.path.join(LANGDATA_FOLDER, name)) and name.endswith(".txt")]

        print("Language files :{}".format(language_corpus))
        print("Creating languages profiles.")

        for language in language_corpus:

            language_name = language.split(".txt")[0]
            ngram_profile, ngram_freq = _ngram_profile(language)
            ngram_language_profiles[language_name] = ngram_profile
            dic_language_profiles[language_name] = [[a,b] for a,b in zip(ngram_profile,ngram_freq)]
            dic_language_profiles[language_name].sort(key=lambda x: x[1], reverse=True)

        print("Languages profiles have been generated successfully")
        print("Languages profiles are going to be stored in {} /language_profile.csv ". format(LANGDATA_PROFILE))

        df = pd.DataFrame.from_dict(dic_language_profiles)
        df.to_csv(LANGDATA_PROFILE + "/" + "language_profile.csv", index=False, encoding='utf-8')

    return ngram_language_profiles


def _ngram_profile(language):
    file_path = LANGDATA_FOLDER + "/" + language

    print("Generating language profile for:{}".format(language))

    corpus_categorizer = ngramfreq.NgramProfileGenerator()

    raw_text = open(file_path, "r", encoding="utf-8").read()
    language_profile = corpus_categorizer.generate_ngram_frequency_profile(raw_text)
    ngram_profile = []
    ngram_frec = []

    for ngram in language_profile:
        ngram_profile.append(ngram[0])
        ngram_frec.append(ngram[1])

    #ngram_profile = [ngram[0] for ngram in language_profile]

    print("Language profile generated")

    #return ngram_profile
    return ngram_profile, ngram_frec
        

#####################
#   PROGRAM BLOCK   #
#####################

# Get language profiles
language_profiles = get_language_profiles()
detection = ngramfreq.LanguageClassifier(language_profiles)


# Confu sion matrix

matriz_confusion, dfallos = detection.guess_language_by_row(TEST_PATH)


# Prediction output
cm = ConfusionMatrix(matriz_confusion['idioma'], matriz_confusion['idioma_pred'])
cm.print_stats()


# Treatment of errors with retraining.

# 1. Join ngrams with it frequences.

df = pd.read_csv(LANGDATA_PROFILE + "/" + "language_profile.csv")
        #pd.to_numeric(df['English'].str.extract('([0-9]\w{0,})', expand=True)[0]) Saca la frecuencia
        #df['English'].str.extract('(\w{0,}[a-z]\w{0,})', expand=True) Saca los n-gramas
dic2 = {}.fromkeys(dfallos.keys())
for i in df.keys():
    a = df[i].str.extract('(\w{0,}[a-z]\w{0,})', expand=True)[0]
    b = pd.to_numeric(df[i].str.extract('([0-9]\w{0,})', expand=True)[0])  # Get ngrams
    dic2[i] = [[x, y] for x, y in zip(a, b)]

# 2. Ngrams of each sentence

for idioma, frases in dfallos.items():
    for i in frases:
        # 3. Search language ngrams and add a unit in each repeated ngram
        tratamiento = ngramfreq.NgramProfileGenerator().generate_ngram_frequency_profile(i)
        for j in tratamiento:
            lst = [item[0] for item in dic2[idioma]]
            if j[0] in lst:
                dic2[idioma][lst.index(j[0])][1] += j[1]
    # 4. List sort before save it
    dic2[idioma].sort(key=lambda x: x[1], reverse=True)

# 5. Save new csv

df2 = pd.DataFrame.from_dict(dic2)
df2.to_csv(LANGDATA_PROFILE + "/" + "language_profile.csv", index=False, encoding='utf-8')

