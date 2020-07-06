# Using callbacks

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


