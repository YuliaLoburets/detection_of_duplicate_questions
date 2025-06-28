import pandas as pd
import numpy as np
from google.colab import drive
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.base import TransformerMixin, BaseEstimator
import networkx as nx
from sentence_transformers import SentenceTransformer
import os

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


stopword_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

class DataTransformer(BaseEstimator, TransformerMixin):
  """
    Preprocesses question pairs by:
    - Lowercasing
    - Removing URLs and punctuation
    - Tokenizing, removing stopwords, and lemmatizing

    Adds 'q1_prep' and 'q2_prep' columns with cleaned text.
  """
  def __init__(self):
    pass

  def preprocess(self, text):
    text = str(text).lower() #make lower text
    text = re.sub(r'http[s]?://\S+', '', text) #remove urls
    text = re.sub(r"[^\w\s]", "", text) #remove punctuation marks
    tokens = nltk.word_tokenize(text) #tokenise text
    text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopword_list] #lemmatize text
    return " ".join(text)

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()
    X['q1_prep'] = X['question1'].apply(self.preprocess)
    X['q2_prep'] = X['question2'].apply(self.preprocess)
    return X

class FeatureCreation(BaseEstimator, TransformerMixin):
  """
  Extracts features from preprocessed question pairs for duplicate detection.

  Features include:
  - TF-IDF and bigram vector similarities
  - Sentence embedding similarity
  - Length, character, and word-based comparisons
  - Jaccard similarity for bigrams/trigrams
  - Cosine and Euclidean distances

  Returns a DataFrame with numeric features for model training.
  """
  def __init__(self):
    self.vectorizer = TfidfVectorizer(max_features=5000)
    self.bigrams_vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english')
    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


  def fit(self, X, y=None):
    all_text = X['q1_prep'].tolist() + X['q2_prep'].tolist()
    self.vectorizer.fit(all_text)
    self.bigrams_vectorizer.fit(all_text)
    return self

  def compute_embeddings_batch(self, texts, batch_size=32):
      embeddings = []
      for i in range(0, len(texts), batch_size):
          batch = texts[i:i+batch_size]
          emb = self.embedding_model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
          embeddings.extend(emb)
      return np.array(embeddings)

  def transform(self, X, y=None):
    X = X.copy()
    vect_q1 = self.vectorizer.transform(X['q1_prep'])
    vect_q2 = self.vectorizer.transform(X['q2_prep'])

    #Sentence embeddings
    emb_q1 = self.compute_embeddings_batch(X['q1_prep'].tolist())
    emb_q2 = self.compute_embeddings_batch(X['q2_prep'].tolist())

    #Semantic cosine similarity
    norm1 = emb_q1 / np.linalg.norm(emb_q1, axis=1, keepdims=True)
    norm2 = emb_q2 / np.linalg.norm(emb_q2, axis=1, keepdims=True)
    X['semantic_sim'] = np.sum(norm1 * norm2, axis=1)

    count_vect_q1 = self.bigrams_vectorizer.transform(X['q1_prep'])
    count_vect_q2 = self.bigrams_vectorizer.transform(X['q2_prep'])

    #determine the number of words in questions
    X['q1_len'] = X['q1_prep'].apply(lambda x: len(x.split()))
    X['q2_len'] = X['q2_prep'].apply(lambda x: len(x.split()))

    #the difference in number of words
    X['length_word_diff'] = X.apply(lambda row: abs(len(row['q1_prep'].split()) - len(row['q2_prep'].split())), axis=1)

    #the word ratio
    X['word_ratio'] = X.apply(lambda row: min(row['q1_len'], row['q2_len']) / (max(row['q1_len'], row['q2_len'])+1e-6), axis=1)

    #the difference in number of characters
    X['q_char_diff'] = X.apply(lambda row: len(row['q1_prep']) - len(row['q2_prep']), axis=1)

    #determine if the questions start with the same word
    X['same_word'] =  X.apply(lambda row: (row['q1_prep'].split()[0].lower() if row['q1_len']>=1 else None) == (row['q2_prep'].split()[0].lower() if row['q2_len']>=1 else None), axis=1).astype('int')

    #number of common bigrams
    X['n_bigrams'] = (count_vect_q1.multiply(count_vect_q2)).sum(axis=1).A1

    #calculate the number of common words in questions
    def word_overlap(row):
      q1 = set(row['q1_prep'].split())
      q2 = set(row['q2_prep'].split())

      return len(q1 & q2), len(q1 & q2) / (len(q1 | q2) + 1e+6)

    overlap = X.apply(word_overlap, axis=1)
    X['number_common_words'] = [result[0] for result in overlap]
    X['ratio_of_common_words'] = [result[1] for result in overlap]

    #cosine similiarity
    cos_sim=[cosine_similarity(q1, q2)[0][0] for q1, q2 in zip(vect_q1, vect_q2)]
    X['cos_sim'] = cos_sim

    #diffence of TD-IF vectors
    X['tdif_diff'] = np.abs(vect_q1-vect_q2).mean(axis=1).A1

    #the strongest common feature in questions
    X['max_similar_word_tdif'] = (vect_q1.multiply(vect_q2)).max(axis=1).toarray().ravel()

    #jaccard bigrams and trigrams
    def get_ngrams(text, n):
      tokens = text.split()
      return set(ngrams(tokens,n))

    def jaccard_ngrams(row,n):
      q1_ngram = get_ngrams(row['q1_prep'], n)
      q2_ngram = get_ngrams(row['q2_prep'], n)
      inter_ngram = set(q1_ngram & q2_ngram)
      all_ngrams = q1_ngram | q2_ngram
      return len(inter_ngram) / (len(all_ngrams) + 1e+6)

    X['jaccard_bigrams'] = X.apply(lambda row: jaccard_ngrams(row,2), axis=1)
    X['jaccard_trigrams'] = X.apply(lambda row: jaccard_ngrams(row,3), axis=1)

    #euclidean distance between vectors
    diff = vect_q1 - vect_q2
    X['euclidean_dist'] = np.sqrt(diff.multiply(diff).sum(axis=1)).A1

    return X[['length_word_diff', 'word_ratio',
              'q_char_diff', 'same_word', 'n_bigrams',
              'number_common_words',  'ratio_of_common_words',
              'cos_sim', 'tdif_diff', 'max_similar_word_tdif',
              'jaccard_bigrams', 'jaccard_trigrams','semantic_sim','euclidean_dist']]
