# Import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

# Create tokenizer
tokenizer = Tokenizer()

# Load data
data = open("Data/irish-lyrics-eof.txt").read()
corpus = data.lower().split("\n")
tokenizer.fit_on_texts()



# In Embedding - Second Value - Bi-directinality can be changed

# LSTM - numeric values can be changes

# Bi directionality can also been removed

# Own Adam optimizer used - different convergences creates different poetry

# More epochs is generally better

