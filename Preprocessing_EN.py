import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pyterrier as pt


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")



class Preprocessing_EN:
  @staticmethod
  def process(sentance):
    sentance = Preprocessing_EN.remove_punctuation(sentance)
    sentance = Preprocessing_EN.tokenizer(sentance)
    sentance = Preprocessing_EN.normalizer(sentance)
    sentance = Preprocessing_EN.remove_stopwords(sentance)
    sentance = Preprocessing_EN.remove_deplicate(sentance)
   

    sentance = Preprocessing_EN.stemmer(sentance)
    return sentance

  @staticmethod
  def tokenizer(sentance):
    words = word_tokenize(sentance)
    return words

  @staticmethod
  def normalizer(sentance):
    return [word.lower() for word in sentance]

  @staticmethod
  def remove_stopwords(sentance):
    stop_words = set(stopwords.words('english'))
    sentance = [word for word in sentance if word not in stop_words]
    return sentance
  
  @staticmethod
  def stemmer(sentance):
    stemmer = PorterStemmer()
    sentance = [stemmer.stem(word) for word in sentance]
    return sentance

  @staticmethod
  def remove_punctuation(sentance):
    return re.sub(r'[^A-Za-z0-9\s]',' ',sentance)

  @staticmethod
  def remove_deplicate(sentance):
    return list(set(sentance))