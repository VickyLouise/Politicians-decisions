import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

def ie_preprocess(document):
    try:
        document = ' '.join([i for i in document.split() if i not in stop])
        sentences = nltk.sent_tokenize(document)
        sentences_words = [nltk.word_tokenize(sent) for sent in sentences]
        sentences_pos = [nltk.pos_tag(sent) for sent in sentences_words]
    except:
        sentences_pos = ''
    return sentences_pos

def get_pos(list_documents):
    list_sentences_pos = []
    for doc in list_documents:
        sentences_pos = ie_preprocess(doc)
        list_sentences_pos.append(sentences_pos)
    return list_sentences_pos

from nltk.corpus import wordnet

def get_wordnet_pos(pos_tuple):
    if pos_tuple[1].startswith('J'):
        return wordnet.ADJ
    elif pos_tuple[1].startswith('V'):
        return wordnet.VERB
    elif pos_tuple[1].startswith('N'):
        return wordnet.NOUN
    elif pos_tuple[1].startswith('R'):
        return wordnet.ADV
    else:
        return None

# Lemmatize the documents (words tagged with parts of speech to improve lemmatization)
import re
from nltk.stem import WordNetLemmatizer

def lemmatize(list_sentences_pos):
    pattern = re.compile('[\w]+')
    lemmatized_dict = {}
    lemmer=WordNetLemmatizer()

    lemmatized_corpus = []
    for doc in list_sentences_pos:
        #print("Document: ", doc)
        lemmatized_document = []
        for sent in doc:
            #print("Sentence: ", sent)
            lemmatized_sentence = []
            for word_pos in sent:
                #print("Word POS: ", word_pos)
                try:
                    # Include the POS for better lemmatisation
                    wntag = get_wordnet_pos(word_pos)
                    if (type(word_pos[0]) == str) and (re.fullmatch(pattern, word_pos[0])):
                        if wntag is None:
                            lemmatized = lemmer.lemmatize(word_pos[0])
                        else:
                            lemmatized = lemmer.lemmatize(word_pos[0],pos=wntag)
                    else:
                        lemmatized = ''
                except(IndexError):
                    lemmatized = ''
                #print("Lemmatized: ", lemmatized)
                lemmatized_sentence.append(lemmatized)
            lemmatized_sentences_all = ' '.join(lemmatized_sentence)
            #print("Lemmatized sentence: ", lemmatized_sentences_all)
            lemmatized_document.append(lemmatized_sentences_all)
        lemmatized_document_all = ''.join(lemmatized_document)   
        #print("Lemmatized document: ", lemmatized_document_all)
        lemmatized_corpus.append(lemmatized_document_all)
    return lemmatized_corpus

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def make_tfidf(lemmatized_corpus):
    tfidf = TfidfVectorizer(lowercase = True,
                            stop_words = "english",
                            ngram_range=(1,2),
                            token_pattern="\\b[a-zA-Z][a-zA-Z]+\\b", #words with >= 2 alpha chars 
                            min_df=0.0075,
                           max_df=0.8,
                           max_features=5000)
    tfidf_vecs = tfidf.fit_transform(lemmatized_corpus)
    df_tfidf = pd.DataFrame(tfidf_vecs.todense(), 
                 columns=tfidf.get_feature_names())
    return tfidf_vecs, df_tfidf