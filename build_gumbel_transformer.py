import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Permute,MaxPooling1D, Flatten, LSTM, Bidirectional, GRU, GlobalAveragePooling1D, dot
try:
	import cPickle as pickle 
except:
	import _pickle as pickle

from keras.datasets import imdb
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras.optimizers import RMSprop
from keras import backend as K  
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.backend import floatx, epsilon
import os, itertools, math 
from keras import losses
from build_model import construct_original_network, TextModel
from load_data import Data
from agccnn.data_helpers import create_vocab_set, encode_data
from keras.engine.topology import Layer



def batch_generator(x, select, y, vocab, vocab_size, vocab_check, maxlen, epoch, batch_size=128):
	for j in xrange(0, len(x)*epoch, batch_size):
		i = j % len(x)
		x_sample = x[i:i + batch_size]
		select_sample = select[i:i + batch_size]
		y_sample = y[i:i + batch_size]

		input_data = encode_data(
			x_sample, 
			maxlen, 
			vocab, 
			vocab_size,
			vocab_check)

		yield ([input_data, select_sample], y_sample)


def data_generator(x, select, vocab, vocab_size, vocab_check, maxlen, batch_size=128):
	for i in xrange(0, len(x), batch_size):
		x_sample = x[i:i + batch_size]
		select_sample = select[i:i + batch_size]

		input_data = encode_data(
			x_sample, 
			maxlen, 
			vocab, 
			vocab_size,
			vocab_check)

		yield [input_data, select_sample]


class Concatenate(Layer):
	def __init__(self, **kwargs): 
		super(Concatenate, self).__init__(**kwargs)

	def call(self, inputs):
		input1, input2 = inputs 
		print(input1.get_shape(),input2.get_shape())
		input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim] 
		dim1 = int(input2.get_shape()[1])
		input1 = tf.tile(input1, [1, dim1, 1])
		return tf.concat([input1, input2], axis = -1)

	def compute_output_shape(self, input_shapes):
		input_shape1, input_shape2 = input_shapes
		input_shape = list(input_shape2)
		input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
		input_shape[-2] = int(input_shape[-2])
		return tuple(input_shape)


class Sample_Concrete(Layer):
	def __init__(self, tau0, **kwargs): 
		self.tau0 = tau0
		super(Sample_Concrete, self).__init__(**kwargs)

	def call(self, logits):  
		unif_shape = tf.shape(logits) 
		uniform = tf.random_uniform(shape = unif_shape, 
			minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
			maxval = 1.0)
		gumbel = - K.log(-K.log(uniform))
		noisy_logits = (gumbel + logits)/self.tau0 
		samples = tf.nn.softmax(noisy_logits) 
		threshold = tf.expand_dims(K.max(logits, axis = -1), axis = -1)
		
		print("learning phase:", K.learning_phase())

		discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)
		output = K.in_train_phase(samples, discrete_logits) 
		return output

	def compute_output_shape(self, input_shape):
		return input_shape


class MakeChange(Layer):
	def __init__(self, **kwargs):
		super(MakeChange, self).__init__(**kwargs)

	def call(self, inputs):  
		original_emb, T, selection, embedding_weights = inputs
		selection = tf.cast(selection, tf.float32)
		selection = tf.expand_dims(selection, axis = -1)
		new_emb = tf.matmul(T, embedding_weights)
		return original_emb * (1.0 - selection) + new_emb * selection 
	def compute_output_shape(self, input_shapes):
		input_shape1, input_shape2, input_shape3, input_shape4 = input_shapes
		return input_shape1


class MakeChangeChar(Layer):
	def __init__(self, **kwargs):
		super(MakeChangeChar, self).__init__(**kwargs)

	def call(self, inputs):  
		original_emb, T, selection = inputs
		selection = tf.cast(selection, tf.float32)
		selection = tf.expand_dims(selection, axis = -1)
		return original_emb * (1.0 - selection) + T * selection 

	def compute_output_shape(self, input_shapes):
		input_shape1, input_shape2, input_shape3 = input_shapes
		return input_shape1


def construct_gumbel_selector(X_ph, num_words, embedding_dims, hidden_dims, maxlen, max_words, network_type = 'cnn'):
	if network_type == 'agccnn':
		emb = X_ph 
		from agvdcnn.Layers import construct_named_convbk
		from agvdcnn.utils import get_conv_shape
		num_filters = [64, 64, 64, 64, 128, 256]
		conv = emb
		input_shape = get_conv_shape(conv)
		share = construct_named_convbk(
			conv, 
			input_shape, 
			num_filters[0], 
			'0'
		)
		conv = share
		for i in xrange(1, 3):
			input_shape = get_conv_shape(conv)
			conv = construct_named_convbk(
				conv, 
				input_shape, 
				num_filters[i], 
				str(i)
			)
			conv = MaxPooling1D(
				pool_size=3, 
				strides=1, 
				padding="same")(conv)
		local_info = conv
		conv = share
		for i in xrange(3, 6):
			print(i)
			input_shape = get_conv_shape(conv)
			conv = construct_named_convbk(
				conv, 
				input_shape, 
				num_filters[i], 
				str(i)
			)
			conv = MaxPooling1D(
				pool_size=3, 
				strides=2, 
				padding="same")(conv) 

		net = Flatten(name = 'flat_gumbel')(conv) # [batchsize, seq_len / 8 * num_filters]
		net = Dropout(0.2)(Dense(256, activation='relu', kernel_initializer='he_normal', name = 'dp_gumbel')(net))
		global_info = Dropout(0.2)(Dense(256, activation='relu', kernel_initializer='he_normal', name = 'dp2_gumbel')(net))
		combined = Concatenate()([global_info,local_info])
		net = Dropout(0.2, name = 'new_dropout_2')(combined)
		net = Conv1D(50, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)  
		logits_T = Conv1D(max_words, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net) 

	elif network_type in ['cnn']:
		emb_layer = Embedding(
			num_words, 
			embedding_dims, 
			input_length = maxlen, 
			name = 'emb_gumbel'
		)
		emb = emb_layer(X_ph) #(400, 50)
		kernel_size = 3
		net = Dropout(0.2, name = 'dropout_gumbel')(emb)
		net = emb
		first_layer = Conv1D(
			100, 
			kernel_size, 
			padding='same', 
			activation='relu', 
			strides=1, 
			name = 'conv1_gumbel')(net)  

		filters = 100; kernel_size = 3; hidden_dims = 100  

		# global info
		# we use max pooling:
		net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(first_layer)
		# We add a vanilla hidden layer:
		global_info = Dense(hidden_dims, name = 'new_dense_1', activation='relu')(net_new) 

		# local info
		net = Conv1D(50, kernel_size, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(first_layer) 
		local_info = Conv1D(50, kernel_size, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(net) 
		combined = Concatenate()([global_info,local_info])
		net = Dropout(0.2, name = 'new_dropout_2')(combined)
		net = Conv1D(50, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)  

		logits_T = Conv1D(max_words, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net) 

	elif network_type == "agccnn2":
		emb = X_ph #(1024, 69)
		kernel_size = 3
		net = Dropout(0.2, name = 'dropout_gumbel')(emb)
		net = emb
		first_layer = Conv1D(
			100, 
			kernel_size, 
			padding='same', 
			activation='relu', 
			strides=1, 
			name = 'conv1_gumbel')(net)  

		filters = 100; kernel_size = 3; hidden_dims = 100  

		# global info
		# we use max pooling:
		net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(first_layer)
		# We add a vanilla hidden layer:
		global_info = Dense(hidden_dims, name = 'new_dense_1', activation='relu')(net_new) 

		# local info
		net = Conv1D(
			50, 
			kernel_size, 
			padding='same', 
			activation='relu', 
			strides=1, 
			name = 'conv2_gumbel')(first_layer) 
		local_info = Conv1D(
			50, 
			kernel_size, 
			padding='same', 
			activation='relu', 
			strides=1, 
			name = 'conv3_gumbel')(net) 
		combined = Concatenate()([global_info,local_info])
		net = Dropout(0.2, name = 'new_dropout_2')(combined)
		net = Conv1D(50, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)  

		logits_T = Conv1D(max_words, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net) 

	elif network_type == 'lstm':
		net = Dropout(0.2, name = 'dropout_gumbel')(emb)
		net = Bidirectional(LSTM(128, return_sequences=True, name = 'lstm_gumbel'), name = 'bidirectional_gumbel')(net)
		logits_T = Conv1D(max_words, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net)
	return logits_T


def negative_xentropy(y_true, y_pred):
	return - K.categorical_crossentropy(y_true, y_pred)


class Gumbel_Transform():
	def __init__(self, data, num_feats, max_words, method, train = False, load_original = False, masking = False):
		self.k = num_feats
		self.maxlen = 400
		self.max_words = max_words
		if data == 'imdbcnn':
			self.num_words = 20002
			embedding_dims = 50
			maxlen = 400
			hidden_dims = 250 
			weights_name = "original.h5"
			emb_name = 'embedding_1'
			num_classes = 2
			num_epoch = 5
		elif data in ['yahoolstm']:
			self.num_words = 20001
			embedding_dims = 300
			maxlen = 400
			hidden_dims = 250
			weights_name = "original-0-7.hdf5"
			emb_name = 'embedding'
			num_classes = 10
			num_epoch = 1
		X_ph = Input(shape = (maxlen,), dtype='int32') 
		weights_extractor_ph = Input(shape = (max_words,), dtype = 'int32')
		Selected_ph = Input(shape = (maxlen,), dtype = 'float32')

		logits_T = construct_gumbel_selector(X_ph, self.num_words, embedding_dims, 
			hidden_dims, maxlen, max_words, network_type = 'cnn')
		tau = 0.5
		T = Sample_Concrete(tau)(logits_T)  
		batch_size = 40

		emb2_layer = Embedding(
			self.num_words, 
			embedding_dims, 
			input_length=maxlen, 
			name = emb_name, 
			trainable = False
		)
		
		embedding_weights = emb2_layer(weights_extractor_ph) 

		X_emb = emb2_layer(X_ph)

		Xnew_emb = MakeChange()([X_emb, T, Selected_ph, embedding_weights])

		preds = construct_original_network(
			Xnew_emb, 
			data, 
			trainable = False)
		
		if train:
			model = Model(
				inputs=[X_ph, Selected_ph, weights_extractor_ph],
				outputs=preds)

			model.compile(
				loss = negative_xentropy,
				optimizer='RMSprop',
				metrics=['acc']
			)
			
			if load_original:
				print('Loading original models...')
				
				model.load_weights('{}/models/{}'.format(data, weights_name), by_name = True)
				
				if data == 'imdbcnn':
					emb_weights = emb2_layer.get_weights() 
					emb_weights[0][0] = np.zeros(50)
					emb2_layer.set_weights(emb_weights)

			dataset = Data(data, True)	

			if method == 'L2X':
				scores_train = np.load('{}/results/scores-train-{}-{}-original{}-mask{}.npy'.format(data, method, num_feats, load_original, masking))
				scores_val = np.load('{}/results/scores-val-{}-{}-original{}-mask{}.npy'.format(data, method, num_feats, load_original, masking))
				label_train  = np.argmax(dataset.pred_train, axis = 1)
				label_train  = np.eye(num_classes)[label_train]
				training_x = dataset.x_train
				filepath="{}/models/gumbel-change-{}-{}-original{}-mask{}.hdf5".format(data, num_feats, max_words, load_original, masking)

			checkpoint = ModelCheckpoint(
				filepath, 
				monitor='val_acc', 
				verbose=1, 
				save_best_only=True,
				mode='min'
			)
			callbacks_list = [checkpoint] 
			print("data loaded")

			selected_train_index = np.argsort(scores_train, axis = -1)[:, -self.k:] # indices of largest k score. 
			selected_train = np.zeros(scores_train.shape)
			selected_train[np.expand_dims(np.arange(len(scores_train)), axis = -1), selected_train_index] = 1.0
			
			selected_val_index = np.argsort(scores_val, axis = -1)[:, -self.k:] # indices of largest k score. 
			selected_val = np.zeros(scores_val.shape)
			selected_val[np.expand_dims(np.arange(len(scores_val)), axis = -1), selected_val_index] = 1.0
			weights_extractor_value = np.tile([list(range(0,max_words-1)) + [self.num_words-1]], [len(scores_train), 1])
			weights_extractor_value_val = np.tile([list(range(0,max_words-1)) + [self.num_words-1]], [len(scores_val), 1])
			label_val  = np.argmax(dataset.pred_val, axis = 1)
			label_val = np.eye(num_classes)[label_val]

			model.fit(
				[training_x, selected_train, weights_extractor_value], 
				label_train, 
				validation_data=([dataset.x_val, selected_val, weights_extractor_value_val], label_val), 
				callbacks = callbacks_list,
				epochs=num_epoch, 
				batch_size=batch_size
			)
			label_train  = np.argmax(dataset.pred_train, axis = 1)
			label_val  = np.argmax(dataset.pred_val, axis = 1)

		else:
			pred_model = Model([X_ph, Selected_ph, weights_extractor_ph], [T])
			pred_model.compile(loss = negative_xentropy, 
				optimizer='RMSprop', metrics=['acc']) 
			
			weights_name = "{}/models/gumbel-change-{}-{}-original{}-mask{}.hdf5".format(data, num_feats, max_words, load_original, masking)
 
			pred_model.load_weights(weights_name, by_name=True)  
			self.pred_model = pred_model 

	def predict(self, xs, selected): 
		weights_extractor_value = np.tile([list(range(0, self.max_words-1)) + [self.num_words-1]], [len(xs), 1])		
		T_value = self.pred_model.predict([xs, selected, weights_extractor_value])
		T_value = np.argmax(T_value, axis = -1)
		T_value = T_value.reshape([-1, self.maxlen])
		for i in xrange(len(T_value)):
			for j in xrange(len(T_value[i])):
				if T_value[i][j] == self.max_words:
					T_value[i][j] = self.num_words - 1
		Xnews_value = xs * (1 - selected) + T_value * selected
		Xnews_value = Xnews_value.astype(int)
		return Xnews_value

	def evaluate(self, data, xs, selected):
		Xnews_value = self.predict(xs, selected)
		dataset = Data(data, False)	
		ori_model = TextModel(data) 
		preds_ori = ori_model.predict(Xnews_value)
		acc_ori = np.mean(np.argmax(preds_ori, axis = 1) == np.argmax(dataset.pred_val[:125], axis = 1))
		return acc_ori


class Gumbel_Transform_Char():
	def __init__(self, data, num_feats, method, train = False, load_original = False, masking = False, max_words = 69):
		if data == 'agccnn':
			hidden_dims = 250 
			filter_kernels = [7, 7, 3, 3, 3, 3]
			dense_outputs = 1024
			self.charlen = 1014
			self.maxlen = None
			nb_filter = 256
			self.num_classes = 4
			self.vocab, self.reverse_vocab, self.vocab_size, self.vocab_check = create_vocab_set()
			self.embedding_dims = self.vocab_size
			K.set_learning_phase(1 if train else 0)

		self.k = num_feats
		self.maxlen = self.vocab_size
		self.max_words = max_words

		X_ph = Input(shape=(self.charlen, self.vocab_size), name='input', dtype='float32') 
		Selected_ph = Input(shape = (self.charlen, ), name = 'input_select', dtype = 'float32')

		logits_T = construct_gumbel_selector(X_ph, None, None, None, None, max_words, network_type = 'agccnn2') 
		
		tau = 0.5 
		T = Sample_Concrete(tau)(logits_T) 

		batch_size = 40

		Xnew_emb = MakeChangeChar()([X_ph, T, Selected_ph])
		conv = Conv1D(
			filters = nb_filter, 
			kernel_size = filter_kernels[0], 
			padding = 'valid', 
			activation = 'relu', 
			input_shape = (self.charlen, self.vocab_size), 
			trainable = False)(Xnew_emb)
		conv = MaxPooling1D(
			pool_size=3, 
			trainable = False)(conv)
		conv1 = Conv1D(
			filters = nb_filter, 
			kernel_size = filter_kernels[1], 
			padding = 'valid', 
			activation = 'relu', 
			trainable = False)(conv)
		conv1 = MaxPooling1D(
			pool_size=3, 
			trainable = False)(conv1) 
		conv2 = Conv1D(
			filters = nb_filter, 
			kernel_size = filter_kernels[2], 
			padding = 'valid', 
			activation = 'relu', 
			trainable = False)(conv1)
		conv3 = Conv1D(
			filters = nb_filter, 
			kernel_size = filter_kernels[3], 
			padding = 'valid', 
			activation = 'relu', 
			trainable = False)(conv2)
		conv4 = Conv1D(
			filters = nb_filter, 
			kernel_size = filter_kernels[4], 
			padding = 'valid', 
			activation = 'relu', 
			trainable = False)(conv3)
		conv5 = Conv1D(
			filters = nb_filter, 
			kernel_size= filter_kernels[5], 
			padding = 'valid', 
			activation = 'relu', 
			trainable = False)(conv4) 
		conv5 = MaxPooling1D(pool_size=3)(conv5)
		conv5 = Flatten()(conv5)
		#Two dense layers with dropout of .5
		z = Dropout(0.5)(Dense(dense_outputs, activation='relu', trainable = False)(conv5))
		z = Dropout(0.5)(Dense(dense_outputs, activation='relu', trainable = False)(z))
		#Output dense layer with softmax activation
		preds = Dense(self.num_classes, activation='softmax', name='output', trainable = False)(z)
			
		if train:
	
			model = Model(inputs=[X_ph, Selected_ph], outputs=preds)
			print("model constructed")
		
			model.compile(loss = negative_xentropy,
						  optimizer='RMSprop',
						  metrics=['acc'])
	
			if load_original:
				print('Loading original models...')
				model.load_weights('{}/params/crepe_model_weights-15.h5'.format(data), by_name=True)	

			filepath="{}/models/gumbel-change-{}-{}-original{}-mask{}.hdf5".format(data, num_feats, max_words, load_original, masking)
			checkpoint = ModelCheckpoint(
				filepath, 
				monitor='val_acc', 
				verbose=1, 
				save_best_only=True, 
				mode='min'
			)
			callbacks_list = [checkpoint] 
			print("data loaded")
			dataset = Data(data, True)	
			if method == 'L2X':
				scores_train = np.load('{}/results/scores-train-{}-{}-original{}-mask{}.npy'.format(data, method, num_feats, load_original, masking))
				scores_val = np.load('{}/results/scores-val-{}-{}-original{}-mask{}.npy'.format(data, method, num_feats, load_original, masking))

			selected_train_index = np.argsort(scores_train, axis = -1)[:, -self.k:] # indices of largest k score. 
			selected_train = np.zeros(scores_train.shape)
			selected_train[np.expand_dims(np.arange(len(scores_train)), axis = -1), selected_train_index] = 1.0
			
			selected_val_index = np.argsort(scores_val, axis = -1)[:, -self.k:] # indices of largest k score. 
			selected_val = np.zeros(scores_val.shape)
			selected_val[np.expand_dims(np.arange(len(scores_val)), axis = -1), selected_val_index] = 1.0

			label_train  = np.argmax(dataset.pred_train, axis = 1)
			label_val  = np.argmax(dataset.pred_val, axis = 1)
			label_train  = np.eye(self.num_classes)[label_train]
			label_val = np.eye(self.num_classes)[label_val]

			generator_train = batch_generator(
				dataset.x_train, 
				selected_train, 
				label_train, 
				self.vocab, 
				self.vocab_size, 
				self.vocab_check, 
				self.charlen, 
				epoch = 6, 
				batch_size = batch_size
			)
			generator_val = batch_generator(
				dataset.x_val, 
				selected_val, 
				label_val, 
				self.vocab, 
				self.vocab_size, 
				self.vocab_check, 
				self.charlen, 
				epoch = 6, 
				batch_size = batch_size
			)
				
			model.fit_generator(
				generator_train, 
				validation_data = generator_val,
				callbacks = callbacks_list,
				epochs = 5,
				steps_per_epoch = len(label_train) / batch_size,
				validation_steps = len(label_val) / batch_size
			)

		else:
			pred_model = Model([X_ph, Selected_ph], [Xnew_emb])
			pred_model.compile(
				loss = negative_xentropy, 
				optimizer='RMSprop',
				metrics=['acc']
			)  
			weights_name = "{}/models/gumbel-change-{}-{}-original{}-mask{}.hdf5".format(data, num_feats, max_words, load_original, masking)
			pred_model.load_weights(
				weights_name, 
				by_name=True
			)  
			self.pred_model = pred_model 
			print("Done Compilation pred model")

	def predict(self, xs, selected): 
		gen_x = data_generator(
			xs, 
			selected, 
			self.vocab, 
			self.vocab_size, 
			self.vocab_check, 
			self.charlen, 
			batch_size = 1000
		)
		Xnew_emb_val = self.pred_model.predict_generator(
			gen_x, 
			steps=math.ceil(float(len(xs))/1000),
			verbose=True
		)
		return Xnew_emb_val

	def evaluate(self, data, xs, selected):
		Xnews_value = self.predict(xs, selected)
		dataset = Data(data, False)	
		ori_model = TextModel(data) 
		preds_ori = ori_model.predict(Xnews_value)
		acc_ori = np.mean(np.argmax(preds_ori, axis = -1) == np.argmax(dataset.pred_val, axis = -1))
		return acc_ori

	
def newloss(y_true, y_pred):
	return dot(y_true, y_pred)


def _to_tensor(x, dtype):
	"""Convert the input `x` to a tensor of type `dtype`.
	# Arguments
		x: An object to be converted (numpy array, list, tensors).
		dtype: The destination type.
	# Returns
		A tensor.
	"""
	return tf.convert_to_tensor(x, dtype=dtype)


def categorical_crossentropy(target, output):  
	_epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)

	output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
	return - tf.reduce_sum(target * tf.log(output))


def prob_to_discrete(probs):
	label = np.zeros(probs.shape)
	idx = np.argmin(probs)# [1,]
	label[idx] = 1 # [?, 400, 500]
	return label


def prob_to_equal(probs):
	label = np.zeros(probs.shape)
	idxes = np.argsort(probs)[:3]
	for idx in idxes:
		label[idx] = 1.0/3
	return label


def probs_generator(x, score, prob, max_words, epoch, batch_size, K = 10):
	for j in xrange(0, len(x)*epoch, batch_size):
		i = j % len(x)
		x_sample = x[i:i + batch_size]
		score_sample = score[i:i + batch_size]
		prob_sample = prob[i:i + batch_size] #(batch_size, 10, 500)
		label_sample = np.zeros((batch_size, 400, max_words))

		for ii in xrange(batch_size):
			for k in xrange(1, K+1):
				selected_index = np.argsort(score_sample[ii], axis = -1)[-k] # indices of largest k score. 
				label_sample[ii, selected_index, :] = prob_to_equal(prob_sample[ii, k-1, :]) #(500, )
		yield (x_sample, label_sample)


def x_generator(x, batch_size):
	for i in xrange(0, len(x), batch_size):
		x_sample = x[i:i + batch_size]
		yield [x_sample]


class Supervised_Transform():
	def __init__(self, data, num_feats, max_words, method, train = False):
		self.k = num_feats
		self.maxlen = 400
		self.max_words = max_words
		if data == 'imdbcnn':
			self.num_words = 20002
			embedding_dims = 50
			maxlen = 400
			hidden_dims = 250 
			weights_name = "original.h5"
			emb_name = 'embedding_1'
			num_classes = 2

		elif data in ['yahoolstm']:
			self.num_words = 20001
			embedding_dims = 300
			maxlen = 400
			hidden_dims = 250
			weights_name = "original-0-7.hdf5"
			emb_name = 'embedding'
			num_classes = 10

		X_ph = Input(shape = (maxlen,), dtype='int32') 
		logits_T = construct_gumbel_selector(
			X_ph, 
			self.num_words, 
			embedding_dims, 
			hidden_dims, 
			maxlen, 
			max_words, 
			network_type = 'cnn'
		)

		preds = Activation('softmax')(logits_T)
		batch_size = 40	
		if train:
			model = Model(inputs=[X_ph], outputs= preds)
			model.compile(
				loss=categorical_crossentropy,
				optimizer='RMSprop',
				metrics=['acc'])

			filepath="{}/models/sup-change-{}-{}.hdf5".format(data, num_feats, max_words)
			checkpoint = ModelCheckpoint(
				filepath, 
				monitor='val_loss', 
				verbose=1, 
				save_best_only=True, 
				mode='min'
			)
			callbacks_list = [checkpoint] 
			print("start data loading")
			dataset = Data(data, True)	

			probs_train = np.load('{}/results/probs_train-jb-leave_one_out.npy'.format(data))
			probs_val = np.load('{}/results/probs_val-jb-leave_one_out.npy'.format(data))

			scores_train = np.load('{}/results/scores-train-{}-{}-original{}-mask{}.npy'.format(data, method, num_feats, 'True', 'True'))
			scores_val = np.load('{}/results/scores-val-{}-{}-original{}-mask{}.npy'.format(data, method, num_feats, 'True', 'True'))

			train_no = 45000
			generator_train = probs_generator(
				dataset.x_train[:train_no], 
				scores_train[:train_no], 
				probs_train[:train_no], 
				max_words, 
				epoch = 10, 
				batch_size = batch_size, 
				K = num_feats
			)
			generator_val = probs_generator(
				dataset.x_val, 
				scores_val, 
				probs_val, 
				max_words, 
				epoch = 10, 
				batch_size = batch_size, 
				K = num_feats
			)
			model.fit_generator(
				generator_train, 
				validation_data = generator_val,
				callbacks = callbacks_list,
				epochs = 10,
				steps_per_epoch = math.ceil(float(train_no)/batch_size),
				validation_steps = math.ceil(float(len(dataset.x_val))/batch_size))

		else:
			pred_model = Model([X_ph], logits_T)
			pred_model.compile(loss = categorical_crossentropy, 
				optimizer='RMSprop', metrics=['acc']) 
			weights_name = "{}/models/sup-change-{}-{}.hdf5".format(data, num_feats, max_words)
			pred_model.load_weights(weights_name, by_name=True)  
			self.pred_model = pred_model 
			print("Done Compilation pred model")

	def predict(self, xs, selected): 
		gen_x = x_generator(xs, batch_size = 1000)
		T_value = self.pred_model.predict_generator(gen_x, steps=math.ceil(float(len(xs))/1000),verbose=True)
		T_value = np.argmax(T_value, axis = -1) #(?, 400)
		T_value = T_value.reshape([-1, self.maxlen])
		for i in xrange(len(T_value)):
			for j in xrange(len(T_value[i])):
				if T_value[i][j] == self.max_words:
					T_value[i][j] = self.num_words - 1

		Xnews_value = xs * (1 - selected) + T_value * selected
		Xnews_value = Xnews_value.astype(int)
		return Xnews_value

	def evaluate(self, data, xs, selected):
		Xnews_value = self.predict(xs, selected)
		dataset = Data(data, False)	
		ori_model = TextModel(data)
		y = ori_model.predict(xs)
		preds_ori = ori_model.predict(Xnews_value)
		acc_ori = np.mean(np.argmax(preds_ori, axis = 1) == np.argmax(dataset.pred_val[:125], axis = 1))
		return acc_ori


		