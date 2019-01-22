import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Permute,MaxPooling1D, Flatten, LSTM, Bidirectional, GRU, GlobalAveragePooling1D
import keras
from keras import backend as K 
from keras.datasets import imdb
from keras.engine.topology import Layer
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import os, itertools, math 
from build_gumbel_transformer import negative_xentropy, construct_gumbel_selector
from build_model import construct_original_network
from agccnn.data_helpers import encode_data, mini_batch_generator, create_vocab_set, construct_batch_generator
import math


class Sample_Concrete(Layer):
	def __init__(self, tau0, k, d, mask, **kwargs): 
		self.tau0 = tau0
		self.k = k
		self.d = d
		self.mask = mask
		super(Sample_Concrete, self).__init__(**kwargs)

	def call(self, logits):   
		# logits: [batch_size, d, 1]
		logits_ = K.permute_dimensions(logits, (0,2,1))# [batch_size, 1, d]
		d = self.d #int(logits_.get_shape()[2])
		unif_shape = tf.shape(logits_)[0]
		uniform = tf.random_uniform(
			shape =(unif_shape, self.k, d), 
			minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
			maxval = 1.0
		)
		gumbel = -K.log(-K.log(uniform))
		noisy_logits = (gumbel + logits_)/self.tau0
		samples = K.softmax(noisy_logits)
		samples = K.max(samples, axis = 1) 
		if self.mask:
			samples = 1.0 - samples
		logits = tf.reshape(logits,[-1, d]) 
		threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
		discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
		if self.mask:
			discrete_logits = 1.0 - discrete_logits
		output = K.in_train_phase(samples, discrete_logits) 
		return tf.expand_dims(output, -1)

	def compute_output_shape(self, input_shape):
		return input_shape


class Gumbel_Selection():
	def __init__(self, num_feats, data, train = False, load_original = False, masking = True):
		if data == 'imdbcnn':
			num_words = 20002
			maxlen = 400
			embedding_dims = 50
			hidden_dims = 250 
			weights_name = "original.h5"
			emb_name = 'embedding_1'
			batch_size = 40
			self.num_classes = 2
			num_epoch = 5
		elif data == 'yahoolstm':
			num_words = 20001
			maxlen = 400
			embedding_dims = 300
			hidden_dims = 250
			weights_name = "original-0-7.hdf5"
			emb_name = 'embedding'
			self.num_classes = 10
			batch_size = 1000
			num_epoch = 1

		Mean = Lambda(
			lambda x: K.sum(x, axis = 1) / float(num_feats), 
			output_shape=lambda x: [x[0],x[2]]
		)

		X_ph = Input(shape=(maxlen,), dtype='int32') 

		logits_T = construct_gumbel_selector(
			X_ph, 
			num_words, 
			embedding_dims, 
			hidden_dims, 
			maxlen, 
			1, 
			network_type = 'cnn'
		)
		tau = 0.5
		sc_layer = Sample_Concrete(tau, num_feats, maxlen, masking)
		T = sc_layer(logits_T) 
		if train:  
			if not load_original:
				filters = 250
				kernel_size = 3
				print('transfer constucted')
				emb_layer = Embedding(
					num_words, 
					embedding_dims, 
					input_length=maxlen, 
					trainable = False
				)
				emb2 = emb_layer(X_ph)
				selected_emb = Multiply()([emb2, T])
				net = Dropout(0.2, trainable = False)(selected_emb)	 
				net = Conv1D(filters,
						kernel_size,
						padding='valid',
						activation='relu',
						strides=1,
						trainable = False)(net)
				net = Dense(hidden_dims, trainable = False)(net)
				net = GlobalMaxPooling1D()(net)
				net = Dense(hidden_dims, trainable = False)(net)
				net = Dropout(0.2, trainable = False)(net)
				net = Activation('relu', trainable = False)(net) 
				net = Dense(self.num_classes, trainable = False)(net)
				preds = Activation('softmax', trainable = False)(net)
				model = Model(inputs=X_ph, outputs=preds)
			else:
				print('original constucted')
				emb_layer = Embedding(
					num_words, 
					embedding_dims, 
					input_length=maxlen, 
					trainable = False
				)
				emb2 = emb_layer(X_ph)
				selected_emb = Multiply()([emb2, T])
				preds = construct_original_network(
					selected_emb, 
					data, 
					trainable = False
				)
				model = Model(inputs=X_ph, outputs=preds)
			
			model.compile(loss= negative_xentropy,
				optimizer='RMSprop',#optimizer,
				metrics=['acc']
			)
					
			if load_original:
				print('Loading original models...')
				model.load_weights(
					'{}/models/{}'.format(data, weights_name),
					by_name = True
				)
			else:
				model.load_weights(
					'{}/models/transfer.hdf5'.format(data), 
					by_name = True
				)

			if data == 'imdbcnn':
				emb_weights = emb_layer.get_weights() 
				emb_weights[0][0] = np.zeros(50)
				emb_layer.set_weights(emb_weights)

			from load_data import Data
			dataset = Data(data, True)	
			
			label_train  = np.argmax(dataset.pred_train, axis = 1)
			label_val  = np.argmax(dataset.pred_val, axis = 1)
			label_val = np.eye(self.num_classes)[label_val]
			label_train  = np.argmax(dataset.pred_train, axis = 1)

			filepath="{}/models/L2X-{}-{}-mask.hdf5".format(data, num_feats, 
				'original' if load_original else 'transfer')

			checkpoint = ModelCheckpoint(
				filepath, 
				monitor='val_acc', 
				verbose=1, 
				save_best_only=True, 
				mode='min'
			)

			callbacks_list = [checkpoint] 

			model.fit(
				dataset.x_train, 
				label_train, 
				validation_data=(dataset.x_val, label_val), 
				callbacks = callbacks_list,
				epochs=num_epoch, 
				batch_size=batch_size
			)

		else:
			pred_model = Model(X_ph, logits_T) 
		  	pred_model.compile(
		  		loss=negative_xentropy, 
				optimizer='RMSprop',
				metrics=['acc']
			)  
		  	weights_name = "{}/models/L2X-{}-{}-mask.hdf5".format(data, num_feats, 'original' if load_original else 'transfer')
			pred_model.load_weights(weights_name, by_name = True)
			self.pred_model = pred_model 

	def predict(self, x): 
		scores = self.pred_model.predict(x, 
			verbose = 1, batch_size = 1000)[:,:,0]
		return scores 


class Gumbel_Selection_Char():
	def __init__(self, num_feats, data, train = False, load_original = False, masking = True): 
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
			#Define what the input shape looks like
			inputs = Input(shape=(self.charlen, self.vocab_size), name='input', dtype='float32')
			logits_T = construct_gumbel_selector(inputs, None, None, None, None, 1, network_type = 'agccnn2')
			# (?, self.charlen, 1)
			tau = 0.5 
			T = Sample_Concrete(tau, num_feats, self.charlen, masking)(logits_T) 
			# (?, self.charlen, 1)
			if train:  
				batch_size = 40
				if not load_original:
					selected_emb = Multiply()([inputs, T]) #(?, self.charlen, 69)

					Mean = Lambda(
						lambda x: K.sum(x, axis = 1) / float(maxlen),
						output_shape=lambda x: [x[0],x[2]]
					)
					emb2 = Embedding(
						self.vocab_size, 
						embedding_dims, 
						input_length=maxlen)(X_ph)
					net = Mean(emb2)
					net = Dense(hidden_dims)(net)
					net = Activation('relu')(net) 
					preds = Dense(2, activation='softmax', 
						name = 'new_dense')(net)
					model = Model(inputs=X_ph, outputs=preds)

				else:
					selected_emb = Multiply()([inputs, T]) #(?, self.charlen, 69)
					conv = Conv1D(
						filters = nb_filter, 
						kernel_size= filter_kernels[0], 
						padding = 'valid', 
						activation = 'relu', 
						input_shape=(self.charlen, self.vocab_size), 
						trainable = False)(selected_emb)
					conv = MaxPooling1D(
						pool_size=3, 
						trainable = False)(conv)
					conv1 = Conv1D(
						filters = nb_filter, 
						kernel_size= filter_kernels[1], 
						padding = 'valid', 
						activation = 'relu', 
						trainable = False)(conv)
					conv1 = MaxPooling1D(
						pool_size=3, 
						trainable = False)(conv1) 
					conv2 = Conv1D(
						filters = nb_filter, 
						kernel_size= filter_kernels[2], 
						padding = 'valid', 
						activation = 'relu', 
						trainable = False)(conv1)
					conv3 = Conv1D(
						filters = nb_filter, 
						kernel_size= filter_kernels[3], 
						padding = 'valid', 
						activation = 'relu', 
						trainable = False)(conv2)
					conv4 = Conv1D(
						filters = nb_filter, 
						kernel_size= filter_kernels[4], 
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
					z = Dropout(0.5)(Dense(
						dense_outputs,
						activation='relu', 
						trainable = False)(conv5))
					z = Dropout(0.5)(Dense(dense_outputs, activation='relu', trainable = False)(z))
					#Output dense layer with softmax activation
					preds = Dense(
						self.num_classes, 
						activation='softmax', 
						name='output', 
						trainable = False)(z)
					model = Model(inputs=inputs, outputs=preds)
				if masking:
					model.compile(loss= negative_xentropy,
								  optimizer='RMSprop',#optimizer,
								  metrics=['acc'])

				print('Loading original models...')
				if load_original:
					model.load_weights('{}/params/crepe_model_weights-15.h5'.format(data), by_name=True)	
				
				if not masking:
					filepath="{}/models/L2X-{}-{}.hdf5".format(data, num_feats, 
						'original' if load_original else 'variational')

					checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
						verbose=1, save_best_only=True, mode='max')
				else:
					filepath="{}/models/L2X-{}-{}-mask.hdf5".format(data, num_feats, 
						'original' if load_original else 'variational')

					checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
						verbose=1, save_best_only=True, mode='min')

				callbacks_list = [checkpoint] 

				from load_data import Data
				dataset = Data(data, True)	
				
				label_train  = np.argmax(dataset.pred_train, axis = 1)
				label_val  = np.argmax(dataset.pred_val, axis = 1)
				label_train  = np.eye(self.num_classes)[label_train]
				label_val = np.eye(self.num_classes)[label_val]

				generator_train = mini_batch_generator(dataset.x_train, label_train, self.vocab, self.vocab_size, self.vocab_check, self.charlen, epoch = 6, batch_size = batch_size)
				generator_val = mini_batch_generator(dataset.x_val, label_val, self.vocab, self.vocab_size, self.vocab_check, self.charlen, epoch = 6, batch_size = batch_size)
				
				model.fit_generator(generator_train, 
					validation_data = generator_val,
					callbacks = callbacks_list,
					epochs = 5,
					steps_per_epoch = len(label_train) / batch_size,
					validation_steps = math.ceil(float(len(label_val))/batch_size),
					verbose=True)
			else:
				pred_model = Model(inputs, logits_T) 
				if not masking:
					pred_model.compile(loss='categorical_crossentropy', 
						optimizer='adam', metrics=['acc'])  

					weights_name = "{}/models/L2X-{}-{}.hdf5".format(data, num_feats, 
						'original' if load_original else 'variational')

				else:
				  	pred_model.compile(loss=negative_xentropy, 
						optimizer='adam', metrics=['acc'])  

					weights_name = "{}/models/L2X-{}-{}-mask.hdf5".format(data, num_feats, 'original' if load_original else 'variational')
				pred_model.load_weights(weights_name, by_name=True)
				self.pred_model = pred_model 

	def predict(self, x): 
		gen_x = construct_batch_generator(x, self.vocab, self.vocab_size, self.vocab_check, self.charlen, batchsize = 1000)
		scores = self.pred_model.predict_generator(gen_x, steps=math.ceil(float(len(x))/1000),verbose=True) 
		scores = np.reshape(scores, [-1, self.charlen])	
		return scores 
