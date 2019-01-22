import pandas as pd 
import numpy as np 
from keras.datasets import imdb
try:
	import cPickle as pickle 
except:
	import _pickle as pickle
import os 
import csv
from load_data import Data 
from keras.utils.data_utils import get_file
import gzip
from keras.preprocessing import sequence 
from six.moves import zip
import numpy as np
import sys
from agccnn.data_helpers import encode_data
from build_model import TextModel 
from load_data import Data 
from keras.utils.np_utils import to_categorical 

def getScore(model, data_sample, pred_sample, positions, classes):
	scores = []
	for i, sample in enumerate(data_sample):
		print('explaining the {}th sample...'.format(i))
		if model.type == 'word': 
			probs = model.predict(np.expand_dims(sample, 0) * positions) # [d, 2] 
		elif model.type == 'char': 
			words = encode_data(
				[sample], 
				model.charlen, 
				model.vocab, 
				model.vocab_size, 
				model.vocab_check
			)
			d = words.shape[1]
			positions = np.expand_dims(1 - to_categorical(range(d), num_classes=d), axis = -1)  # (d, d)
			probs = model.predict(words * positions)

		score = pred_sample[i, classes[i]] - probs[:, classes[i]]
		scores.append(score)
	return scores



 
