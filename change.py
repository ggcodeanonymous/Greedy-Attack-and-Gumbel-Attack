from __future__ import absolute_import, division, print_function 
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Permute, Dot
from keras.datasets import imdb
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras import backend as K  
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
import os
import time 
import sys
from agccnn.data_helpers import encode_data
from build_model import TextModel 
from load_data import Data 
try:
	import cPickle as pickle 
except:
	import _pickle as pickle


# set parameters:
max_features = 500
maxlen = 400
batch_size = 40
embedding_dims = 50       
filters = 250
kernel_size = 3
hidden_dims = 250
PART_SIZE = 160
epochs = 5
k =10
max_words = 100
MAX_NB_WORDS = 20000


def greedy_change_the_word_K(score, model, x, original_prob, K = 10, sign = True):
	"""
	selected_points: a list of length K, with each element of the same dimension of x. 
					selected_points[k]: we seletct k indexes of x to change for k = 1, ..., K
					selected_points[k][i] = 1 if i is the index we select to change, 0 otherwise
	changed_sequences: a list of length K, with each element of the same dimension of x. 
					changed_sequences[k]: the changed sequence of x with k elements changed for k = 1, ..., K

	when k == 0: we perturb the sequence until its predicted label is changed.
	"""
	if not sign:
		score = abs(score)

	selected_points = []
	changed_x = []
	nums = np.arange(1, K+1)
	if args.data == 'agccnn':
		word = encode_data([x], model.charlen, model.vocab, model.vocab_size, model.vocab_check)[0]
		changed_sequence = word
	else:
		changed_sequence = x
	original_y = np.argmax(original_prob)

	if K == 0:
		current_label = original_y
		k = 0
		while current_label == original_y:
			k += 1
			selected = np.argsort(score)[-k:] # indices of largest k score. 
			d = len(score)
			selected_k_hot = np.zeros(d)
			selected_k_hot[selected] = 1.0
			if args.data == 'imdbcnn':
				# selected_points.append(x * selected_k_hot) # selecting largest k. 
				# find the index of the kth change 
				selected_k = np.argsort(score)[-k]
				# make the change
				changed_sequences = np.tile(np.expand_dims(changed_sequence, axis = 0),[max_features, 1]) #[max_features, d]
				changed_sequences[:, selected_k] = np.arange(1, max_features+1)

				probs_changed_sequences = model.predict(changed_sequences)
				# index of which change_sequence we want
				index_sequence = np.argmin(probs_changed_sequences[:, original_y])
				
				changed_sequence = changed_sequences[index_sequence]

				current_label = np.argmax(probs_changed_sequences[index_sequence])

		selected_points.append(x * selected_k_hot) # selecting largest k. 
		changed_x.append(changed_sequence)
		return selected_points, changed_x

	for k in nums:
		selected = np.argsort(score)[-k:] # indices of largest k score. 
		d = len(score)
		selected_k_hot = np.zeros(d)
		selected_k_hot[selected] = 1.0
		if args.data != 'agccnn':
			selected_points.append(x * selected_k_hot) # selecting largest k. 
			# find the index of the kth change 
			selected_k = np.argsort(score)[-k]
			# make the change
			changed_sequences = np.tile(np.expand_dims(changed_sequence, axis = 0),[max_features, 1]) #[max_features, d]
			changed_sequences[:, selected_k] = np.arange(1, max_features+1)

			probs_changed_sequences = model.predict(changed_sequences)
			# index of which change_sequence we want
			index_sequence = np.argmin(probs_changed_sequences[:, original_y])
			
			changed_sequence = changed_sequences[index_sequence]
			changed_x.append(changed_sequence)
		elif args.data == 'agccnn':
			selected_points.append( word * np.expand_dims(selected_k_hot, axis = -1))
			selected_k = np.argsort(score)[-k]
			changed_sequences = np.tile(np.expand_dims(changed_sequence, axis = 0),[model.vocab_size, 1, 1]) #[69, 1014, 69]
			changed_sequences[:, selected_k, :] = np.diag(np.ones(model.vocab_size))
			probs_changed_sequences = model.predict(changed_sequences)
			index_sequence = np.argmin(probs_changed_sequences[:, original_y])
			changed_sequence = changed_sequences[index_sequence]
			changed_x.append(changed_sequence)
	return selected_points, changed_x


def change_with_method(model, scores, xs, dx_s, method, changing_way, sign = True):
	changed_xs = []
	selected_K = []
	for i, score in enumerate(scores):
		x = xs[i]
		prob = model.predict(np.expand_dims(x, axis = 0))
		print("Change data", i)
		if changing_way == "greedy_change_k": 
			selected_points, changed_x = greedy_change_the_word_K(score, model, x, prob, K = args.num_feats, sign = True) 
		changed_xs.append(changed_x)
		selected_K.append(selected_points)
	return selected_K, changed_xs


def change(args): 
	print('Loading dataset...') 
	method, changing_way = args.method, args.changing_way
	dataset, model =  args.dataset, args.model 	
	xs = dataset.x_val
	st = time.time()
	dx_s = None
	if method == 'L2X':	
		scores = np.load('{}/results/scores-{}-{}-original{}-mask{}.npy'.format(args.data, args.method, args.num_feats, args.original, args.mask))
		selected_K, changed_xs = change_with_method(model, scores, xs, dx_s, method, changing_way, sign = True)
		filename = '{}/results/changed_xs-{}-{}-original{}-mask{}.npy'.format(args.data, method, changing_way, args.original, args.mask)
		np.save('{}/results/selected_K-{}-original{}-mask{}.npy'.format(args.data, method, args.original, args.mask), selected_K)
		np.save(filename, changed_xs)		
	else:
		scores = np.load('{}/results/scores-{}.npy'.format(args.data, method))
		selected_K, changed_xs = change_with_method(model, scores, xs, dx_s, method, changing_way, sign = True)
		np.save('{}/results/changed_xs-{}-{}.npy'.format(args.data, method, changing_way), changed_xs)
		np.save('{}/results/selected_K-{}.npy'.format(args.data, method), selected_K)
	print("Done changing x with method", changing_way)
	return selected_K, changed_xs, [time.time() - st]


def change_train(args):
	print('Loading dataset...') 
	method, changing_way = args.method, args.changing_way
	dataset, model =  args.dataset, args.model 	
	xs = dataset.x_train
	dx_s = None
	scores = np.load('{}/results/scores-train-{}.npy'.format(args.data, method))
	selected_K, changed_xs = change_with_method(model, scores, xs, dx_s, method, changing_way, sign = True)
	np.save('{}/results/changed_xs-train-{}-{}.npy'.format(args.data, method, changing_way), changed_xs)
	np.save('{}/results/selected_K-train-{}.npy'.format(args.data, method), selected_K)
	print("done changing x_train with method", changing_way)
	return selected_K, changed_xs


def gumbel(args):
	from build_gumbel_transformer import Gumbel_Transform, Gumbel_Transform_Char
	if args.data == 'agccnn':
		gumbel_transform = Gumbel_Transform_Char(
			args.data, 
			args.num_feats, 
			args.method, 
			args.train, 
			args.original, 
			args.mask
		)
	else:
		gumbel_transform = Gumbel_Transform(
			args.data, 
			args.num_feats, 
			args.max_words, 
			args.method, 
			args.train, 
			args.original, 
			args.mask
		)
	if not args.train:
 		dataset = Data(args.data)
 		if args.method == 'L2X':

 			scores = np.load('{}/results/scores-{}-{}-original{}-mask{}.npy'.format(args.data, args.method, args.num_feats, args.original, args.mask))
 				
 		elif args.method == 'leave_one_out':
 			scores = np.load('{}/results/scores-{}.npy'.format(args.data, args.method))

 		changed_xs = []
 		st = time.time()

 		for k in xrange(1, args.num_feats+1):
			selected_index = np.argsort(scores, axis = -1)[:, -k:] # indices of largest k score. 
			selected = np.zeros(scores.shape)
			selected[np.expand_dims(np.arange(len(scores)), axis = -1), selected_index] = 1.0
 			changed_x = gumbel_transform.predict(dataset.x_val, selected)

 			changed_xs.append(changed_x)
 		changed_xs = np.array(changed_xs)
		changed_xs = np.swapaxes(changed_xs, 0, 1) 
		return changed_xs, [time.time() - st]
	return None, None


def prob_change_x(data, model, x, y, score, K = 10, max_words = 500):
	probs = np.zeros((K, max_words))
	for k in xrange(1, K+1):
		selected_index = np.argsort(score, axis = -1)[-k] # indices of largest k score. 
		changed_x = np.tile(np.expand_dims(x, axis = 0), [max_words, 1]) #(max_words, d)
		changed_x[:, selected_index] = list(np.arange(max_words-1)) + [20002] #(max_words, d) # 
		prob = model.predict(changed_x)[:, y] #(max_words, 1)
		prob = prob.reshape((max_words,))  #(max_words)
		probs[k-1] = prob #(400, max_words)  
	return probs


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--method', 
		type = str, 
		choices = ['leave_one_out', 'L2X', 'create_predictions', 'train'], 
		default = 'leave_one_out'
	)
	parser.add_argument(
		'--data', 
		type = str, 
		choices = ['imdbcnn', 'agccnn', 'yahoolstm'], 
		default = 'imdbcnn'
	) 
	parser.add_argument(
		"--changing_way", 
		type = str, 
		choices = ['greedy_change_k', 'gumbel'], 
		default = 'greedy_change_k'
	)
	parser.add_argument('--num_feats', type = int, default = 10)
	parser.add_argument('--max_words', type = int, default = 69)
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--original', action='store_true')
	parser.add_argument('--mask', action='store_true')
	parser.add_argument('--change_train', action='store_true')

	args = parser.parse_args()
	dict_args = vars(args)   
	if args.method == 'train':
		model = TextModel(args.data, train = True)

	if args.data not in os.listdir('./'):	
		os.mkdir(args.data)
	if 'results' not in os.listdir('./{}'.format(args.data)):
		os.mkdir('{}/results'.format(args.data))

	if args.changing_way == 'gumbel':
		changed_xs, times = gumbel(args)
		np.save('{}/results/changed_xs-{}-{}-{}-original{}-mask{}.npy'.format(args.data, args.method, args.changing_way, args.max_words, args.original, args.mask), changed_xs)
	else:
		dataset = Data(args.data)
		model = TextModel(args.data) 
		dict_args.update({'dataset': dataset, 'model': model})

		if args.change_train:
			selected_K, changed_xs = change_train(args)
		else:
			selected_K, changed_xs, times = change(args)
			np.save('{}/results/time-trans-{}-{}-{}'.format(args.data, args.data, args.method, args.changing_way), times)
		




