from __future__ import absolute_import, division, print_function 
import tensorflow as tf
from keras import backend as K  
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Permute
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical 
from agccnn.data_helpers import encode_data, mini_batch_generator, construct_batch_generator
from build_model import TextModel 
from load_data import Data 
from utils import getScore
import tarfile
import zipfile 
try:
	import cPickle as pickle 
except:
	import _pickle as pickle
import numpy as np
import os
import time 
import h5py
import sys
sys.path.append('..')


def create_original_predictions(args):
	# save original validation prediction probabilities.  
	dataset = Data(args.data, True)
	model = TextModel(args.data, False)
	pred_val = model.predict(dataset.x_val, verbose = True)
	pred_train = model.predict(dataset.x_train, verbose = True)

	if 'data' not in os.listdir(args.data):
		os.mkdir('{}/data'.format(args.data))

	np.save('{}/data/pred_val.npy'.format(args.data), pred_val)	
	np.save('{}/data/pred_train.npy'.format(args.data), pred_train)
		
	acc_val = np.mean(np.argmax(pred_val, axis = 1) == np.argmax(dataset.y_val, axis = 1))
	acc_train = np.mean(np.argmax(pred_train, axis = 1) == np.argmax(dataset.y_train, axis = 1))
	print('The validation accuracy is {}.'.format(acc_val))
	print('The training accuracy is {}.'.format(acc_train))

	if args.data != 'agccnn':
		np.save('{}/data/embedding_matrix.npy'.format(args.data), model.emb_weights)


def leave_one_out(args):
	dataset, model =  args.dataset, args.model 
	st = time.time()
	print('Making explanations...') 
	if model.maxlen: 
		positions = 1 - to_categorical(range(model.maxlen), num_classes=model.maxlen)  

	if model.type == 'word':
		if args.train_score:
			pred_train = model.predict(dataset.x_train)
		else:
			pred_val = model.predict(dataset.x_val)
	elif model.type == 'char':
		if args.train_score:
			pred_train = model.predict(
				encode_data(dataset.x_train, model.charlen, model.vocab, model.vocab_size, model.vocab_check)
			)
		else:
			pred_val = model.predict(
				encode_data(dataset.x_val, model.charlen, model.vocab, model.vocab_size, model.vocab_check)
			)

	classes = np.argmax(pred_train, axis = 1) if args.train_score else np.argmax(pred_val, axis = 1)

	if args.train_score:
		scores = getScore(model, dataset.x_train, pred_train, positions, classes)
	else:
		scores = getScore(model, dataset.x_val, pred_val, positions, classes)
	print('Time spent is {}'.format(time.time() - st))
	return scores, [time.time() - st]


def L2X(args):
	from build_gumbel_selector import Gumbel_Selection, Gumbel_Selection_Char

	if args.data == 'agccnn':
		gumbel_selector = Gumbel_Selection_Char(
			args.num_feats, 
			args.data, 
			args.train, 
			args.original, 
			args.mask
		)
	else:
		gumbel_selector = Gumbel_Selection(
			args.num_feats, 
			args.data, 
			args.train, 
			args.original, 
			args.mask, 
		)
	if args.train:
		return None, None
	else:
		if args.train_score:
			dataset = Data(args.data, True)
			scores_val = gumbel_selector.predict(dataset.x_val)
			np.save('{}/results/scores-val-{}-{}-original{}-mask{}.npy'.format(args.data, args.method, args.num_feats, args.original, args.mask), scores_val)
			scores_train = gumbel_selector.predict(dataset.x_train)
			np.save('{}/results/scores-train-{}-{}-original{}-mask{}.npy'.format(args.data, args.method, args.num_feats, args.original, args.mask), scores_train)
		
		dataset = Data(args.data, False)
		st = time.time()
		scores = gumbel_selector.predict(dataset.x_val)
		print('Time spent is {}'.format(time.time() - st))
		return scores, [time.time() - st]


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--method', 
		type = str, 
		choices = ['leave_one_out', 'L2X', 'training', 'create_predictions']
	)

	parser.add_argument(
		'--data', 
		type = str, 
		choices = ['imdbcnn', 'agccnn','yahoolstm'], 
		default = 'imdbcnn'
	) 

	parser.add_argument(
		'--num_feats', 
		type = int, 
		default = 10
	)

	parser.add_argument(
		'--train', 
		action='store_true'
	)

	parser.add_argument(
		'--train_score', 
		action = 'store_true'
	)

	parser.add_argument(
		'--mask', 
		action='store_true'
	)

	parser.add_argument(
		'--original', 
		action='store_true'
	)

	args = parser.parse_args()
	dict_args = vars(args)   

	if args.method == 'training':
		model = TextModel(args.data, train = True)
	
	elif args.method == 'create_predictions':
		create_original_predictions(args)

	elif args.method == 'L2X': 
		scores, time = L2X(args)
		filename = '{}/results/scores-{}-{}-original{}-mask{}.npy'.format(args.data, args.method, args.num_feats, args.original, args.mask)
		np.save(filename, scores)

	elif args.method == 'leave_one_out':
		print('Loading dataset...') 
		dataset = Data(args.data)
		print('Creating model...')
		model = TextModel(args.data) 
		dict_args.update({'dataset': dataset, 'model': model})
	
		scores, time = leave_one_out(args) 
		np.save('{}/results/scores-{}.npy'.format(args.data, args.method), scores)















