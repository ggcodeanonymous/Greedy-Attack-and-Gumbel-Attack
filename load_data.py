from keras.datasets import imdb
import numpy as np 
from keras.preprocessing import sequence 
try:
	import cPickle as pickle 
except:
	import _pickle as pickle
import os
import sys
from imdbcnn.imdb_data import create_imdb_dataset


class Data():
	def __init__(self, data, train = False, no_samples = 125):
		if data == 'imdbcnn':
			if 'data' not in os.listdir('./imdbcnn'):
				os.mkdir('./imdbcnn/data')
			data_dir = './imdbcnn/data'
			if 'models' not in os.listdir('./imdbcnn'):
				os.mkdir('./imdbcnn/models')
			if 'results' not in os.listdir('./imdbcnn'):
				os.mkdir('./imdbcnn/results')

			if 'pred_val.npy' in os.listdir(data_dir):
				self.pred_train = np.load('{}/pred_train.npy'.format(data_dir))
				self.pred_val = np.load('{}/pred_val.npy'.format(data_dir))
			if 'x_val.npy' in os.listdir(data_dir):
				x_val, y_val = np.load('{}/x_val.npy'.format(data_dir)), np.load('{}/y_val.npy'.format(data_dir))
				x_train, y_train = np.load('{}/x_train.npy'.format(data_dir)), np.load('{}/y_train.npy'.format(data_dir))
			else:
				print('Loading data...')
				data, labels, texts, word_index, data_unlabel = create_imdb_dataset()
				id_to_word = {value: key for key,value in word_index.items()}

				indices = np.arange(data.shape[0])
				np.random.seed(0)
				np.random.shuffle(indices)

				data = data[indices]
				texts = texts[indices]
				labels = labels[indices]
				VALIDATION_SPLIT = 0.1
				nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
				x_train_raw = texts[:-nb_validation_samples]
				x_train = data[:-nb_validation_samples]
				y_train = labels[:-nb_validation_samples]
				x_val_raw = texts[-nb_validation_samples:]
				x_val = data[-nb_validation_samples:]
				y_val = labels[-nb_validation_samples:]

				np.save('{}/x_train.npy'.format(data_dir), x_train)
				np.save('{}/y_train.npy'.format(data_dir), y_train)

				np.save('{}/x_val.npy'.format(data_dir), x_val)
				np.save('{}/y_val.npy'.format(data_dir), y_val)

				np.save('{}/x_train_raw.npy'.format(data_dir), x_train_raw)
				np.save('{}/x_val_raw.npy'.format(data_dir), x_val_raw)

				with open('imdbcnn/data/id_to_word.pkl','wb') as f:
					pickle.dump(id_to_word, f)	

			with open('{}/id_to_word.pkl'.format(data_dir),'rb') as f:
				id_to_word = pickle.load(f)
				self.id_to_word = id_to_word 
			self.x_train = x_train
			self.y_train = y_train
			self.x_val = x_val if train else x_val[:no_samples] 
			self.y_val = y_val if train else y_val[:no_samples] 
			
		elif data == 'agccnn':
			data_dir = './agccnn/data'
			if 'data' not in os.listdir('./agccnn'):
				os.mkdir(data_dir)
			if 'models' not in os.listdir('./agccnn'):
				os.mkdir('./agccnn/models')
			if 'results' not in os.listdir('./agccnn'):
				os.mkdir('./agccnn/results')

			if 'x_val.npy' in os.listdir(data_dir):
				x_val, y_val = np.load('{}/x_val.npy'.format(data_dir)), np.load('{}/y_val.npy'.format(data_dir))
				x_train, y_train = np.load('{}/x_train.npy'.format(data_dir)), np.load('{}/y_train.npy'.format(data_dir))
			else:
				from agccnn.data_helpers import load_ag_data 
				print('Loading data...')
				(x_train, y_train), (x_val, y_val) = load_ag_data('agccnn') 

				indices = range(len(x_val))
				np.random.seed(0)
				np.random.shuffle(indices)
				x_val = x_val[indices]
				y_val = y_val[indices]

				np.save('{}/x_val.npy'.format(data_dir), x_val)
				np.save('{}/y_val.npy'.format(data_dir), y_val) 
				np.save('{}/x_train.npy'.format(data_dir), x_train)
				np.save('{}/y_train.npy'.format(data_dir), y_train) 

			if 'pred_val.npy' in os.listdir(data_dir):
				pred_val = np.load('{}/pred_val.npy'.format(data_dir))
				pred_train = np.load('{}/pred_train.npy'.format(data_dir))
				self.pred_val = pred_val
				self.pred_train = pred_train

			self.x_train = x_train
			self.y_train = y_train
			self.x_val = x_val if train else x_val[:no_samples] 
			self.y_val = y_val if train else y_val[:no_samples] 

		elif data == 'yahoolstm': 

			data_dir = 'yahoolstm/data' 
			if 'models' not in os.listdir('./yahoolstm'):
				os.mkdir('./yahoolstm/models')
			if 'results' not in os.listdir('./yahoolstm'):
				os.mkdir('./yahoolstm/results')

			if 'x_test.npy' not in os.listdir(data_dir):
				print('Processing text dataset...')  
				import csv
				from keras.preprocessing.text import Tokenizer
				from keras.utils import to_categorical 

				sentences, labels = [], []
				for data_type in ['train','test']:
					filename = "yahoolstm/data/{}_new.csv".format(data_type)
					f = open(filename, 'r')
					reader = csv.DictReader(f, fieldnames=['label', 'title','title2', 'description'], quotechar='"')

					for i, line in enumerate(reader): 
						sentence = "{} {} {}".format(line['title'],line['title2'], line['description']) 
						label = int(line['label']) - 1
						sentences.append(sentence)
						labels.append(label)  	 

				MAX_SEQUENCE_LENGTH = 400 
				MAX_NB_WORDS = 20000
				tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token = "<UNK>")
				tokenizer.fit_on_texts(sentences)
				sequences = tokenizer.texts_to_sequences(sentences)
				word_index = tokenizer.word_index
				data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
				labels = to_categorical(np.asarray(labels))
				np.save('{}/x_{}.npy'.format(data_dir, 'train'), data[:1400000])
				np.save('{}/y_{}.npy'.format(data_dir, 'train'), labels[:1400000])
				np.save('{}/x_{}.npy'.format(data_dir, 'test'), data[1400000:])
				np.save('{}/y_{}.npy'.format(data_dir, 'test'), labels[1400000:])

			x_val, y_val = np.load('{}/x_test.npy'.format(data_dir)),np.load('{}/y_test.npy'.format(data_dir))
			if train:
				x_train, y_train = np.load('{}/x_train.npy'.format(data_dir)), np.load('{}/y_train.npy'.format(data_dir))
				self.x_train = x_train
				self.y_train = y_train
			else: 
				np.random.seed(0)
				idx = np.random.permutation(60000)
				x_val, y_val = x_val[idx], y_val[idx]
				x_val, y_val = x_val[:no_samples], y_val[:no_samples]
				val_len = np.load('yahoolstm/data/len_test.npy')
				val_len = np.minimum(val_len, 400)
				val_len = val_len[idx][:no_samples]
				self.val_len = val_len

			self.x_val = x_val 
			self.y_val = y_val 		
				
			data_dir1 = '{}/data'.format(data)
			if 'pred_val.npy' in os.listdir(data_dir1):
				pred_val = np.load('{}/pred_val.npy'.format(data_dir))
				pred_train = np.load('{}/pred_train.npy'.format(data_dir))
				self.pred_val = pred_val
				self.pred_train = pred_train				 

