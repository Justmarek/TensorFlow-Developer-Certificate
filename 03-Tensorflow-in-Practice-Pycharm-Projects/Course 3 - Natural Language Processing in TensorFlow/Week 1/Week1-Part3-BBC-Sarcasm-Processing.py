# Import libraries
import csv
import nltk
from nltk.corpus import stopwords
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import stopwords
stop_words = set(stopwords.words('english'))
stop_words = list(stop_words)

# Import data
datastore = pd.read_csv("data/bbc-text.csv")
labels = datastore["category"].tolist()
sentences = datastore["text"].tolist()
print(len(sentences))
print(sentences[0])

# Instantiate tokenizer and create word index
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))

# Pad sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)