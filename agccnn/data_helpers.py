import string
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


def load_ag_data(dir = '.'):
	train = pd.read_csv('{}/data/ag_news_csv/train.csv'.format(dir), header=None)
	train = train.dropna()

	x_train = train[1] + train[2]
	x_train = np.array(x_train)

	y_train = train[0] - 1
	y_train = to_categorical(y_train)

	test = pd.read_csv('{}/data/ag_news_csv/test.csv'.format(dir), header=None)
	x_test = test[1] + test[2]
	x_test = np.array(x_test)

	y_test = test[0] - 1
	y_test = to_categorical(y_test)
	return (x_train, y_train), (x_test, y_test)


def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen, epoch,
						 batch_size=128):
	for j in xrange(0, len(x)*epoch, batch_size):
		i = j % len(x)
		x_sample = x[i:i + batch_size]
		y_sample = y[i:i + batch_size]

		input_data = encode_data(x_sample, maxlen, vocab, vocab_size,
								 vocab_check)
		yield (input_data, y_sample)


def construct_batch_generator(x, vocab, vocab_size, vocab_check, maxlen,
						 batchsize=128):

	for i in xrange(0, len(x), batchsize):
		x_sample = x[i:i + batchsize] 

		input_data = encode_data(x_sample, maxlen, vocab, vocab_size,
								 vocab_check)

		yield input_data


def encode_data(x, maxlen, vocab, vocab_size, check):
	#Iterate over the loaded data and create a matrix of size maxlen x vocabsize
	#In this case that will be 1014x69. This is then placed in a 3D matrix of size
	#data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
	#array. Chars not in the vocab are encoded into an all zero vector.

	input_data = np.zeros((len(x), maxlen, vocab_size))
	for dix, sent in enumerate(x): 
		counter = 0
		sent_array = np.zeros((maxlen, vocab_size))
		chars = list(sent.lower().replace(' ', ''))
		for c in chars:
			if counter >= maxlen:
				pass
			else:
				char_array = np.zeros(vocab_size, dtype=np.int)
				if c in check:
					ix = vocab[c]
					char_array[ix] = 1
				sent_array[counter, :] = char_array
				counter += 1
		input_data[dix, :, :] = sent_array
	return input_data


def find_words_positions(sent, words, maxlen, vocab, vocab_size, check, char = False):
	# input_vector = encode_data([x], maxlen, vocab, vocab_size, check)
	if char:
		chars = sent.lower()
	else:
		chars = sent.lower().replace(' ', '') 
		
	index = 0
	indices = []

	words = [word.lower() for word in words]
	words = [filter(lambda x: x in check, word) for word in words]

	for word in words:
		start_idx = chars[index:].find(word) 
		start_idx += index 

		end_idx = start_idx + len(word) # The end idx is exclusive.
		indices.append([start_idx, end_idx])

		index = end_idx 
	return indices 


def shuffle_matrix(x, y):
	stacked = np.hstack((np.matrix(x).T, y))
	np.random.shuffle(stacked)
	xi = np.array(stacked[:, 0]).flatten()
	yi = np.array(stacked[:, 1:])
	return xi, yi


def create_vocab_set():
	#This alphabet is 69 chars vs. 70 reported in the paper since they include two
	# '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

	# vocab: a dictionary with key: alphabet and value : index
	alphabet = (list(string.ascii_lowercase) + list(string.digits) +
				list(string.punctuation) + ['\n'])
	vocab_size = len(alphabet)
	check = set(alphabet)

	vocab = {}
	reverse_vocab = {}
	for ix, t in enumerate(alphabet):
		vocab[t] = ix
		reverse_vocab[ix] = t
	return vocab, reverse_vocab, vocab_size, check


