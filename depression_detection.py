import pandas as pd 
import re
import gensim 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import warnings
from  sklearn.metrics import accuracy_score
from sklearn import preprocessing
warnings.filterwarnings("ignore")