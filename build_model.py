import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Permute,MaxPooling1D, Flatten, LSTM, Bidirectional, GRU, GlobalAveragePooling1D
from keras.datasets import imdb
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras.optimizers import RMSprop
from keras import backend as K  
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import os, itertools, math 


def construct_original_network(emb, data,trainable=True):
	if data == 'imdbcnn':
		filters = 250
		kernel_size = 3
		hidden_dims = 250
		net = Dropout(0.2, name = 'dropout_1')(emb)		 
		# we add a Convolution1D, which will learn filters
		# word group filters of size filter_length:
		net = Conv1D(filters,
						 kernel_size,
						 padding='valid',
						 activation='relu',
						 strides=1,
						 name = 'conv1d_1',trainable=trainable)(net)
		# we use max pooling:
		net = GlobalMaxPooling1D(name = 'global_max_pooling1d_1')(net)
		# We add a vanilla hidden layer:
		net = Dense(hidden_dims, name = 'dense_1',trainable=trainable)(net)
		net = Dropout(0.2, name = 'dropout_2')(net)
		net = Activation('relu', name = 'activation_2')(net)
		# We project onto a single unit output layer, and squash it with a sigmoid:
		net = Dense(2, name = 'dense_2',trainable=trainable)(net)
		preds = Activation('softmax', name = 'activation_3')(net)
		return preds
	elif data == 'yahoolstm':
		lstm_out = Bidirectional(LSTM(256,trainable=trainable), trainable = trainable)(emb)  
		net = Dropout(0.5)(lstm_out)
		preds = Dense(10, activation='softmax',trainable=trainable)(net)
		return preds


class TextModel():
	def __init__(self, data, train = False):
		self.data = data
		print('Loading TextModel...')
		if data == 'imdbcnn':
			filters = 250 
			hidden_dims = 250
			self.embedding_dims = 50
			self.maxlen = 400
			self.num_classes = 2
			self.num_words = 20002
			self.type = 'word'
			if not train:
				K.set_learning_phase(0)

			X_ph = Input(shape=(self.maxlen,), dtype='int32')
			emb_layer = Embedding(
				self.num_words, 
				self.embedding_dims,
				input_length=self.maxlen, 
				name = 'embedding_1'
			)
			emb_out = emb_layer(X_ph) 

			if train:
				preds = construct_original_network(emb_out, data)	

			else: 
				emb_ph = Input(
					shape=(self.maxlen, self.embedding_dims), 
					dtype='float32'
				)   
				preds = construct_original_network(emb_ph, data) 

			if not train:
				model1 = Model(X_ph, emb_out)
				model2 = Model(emb_ph, preds) 
				pred_out = model2(model1(X_ph))  
				pred_model = Model(X_ph, pred_out) 
				pred_model.compile(
					loss='categorical_crossentropy',
					optimizer='adam', 
					metrics=['accuracy']
				) 
				self.pred_model = pred_model 
				grads = []
				for c in range(self.num_classes):
					grads.append(tf.gradients(preds[:,c], emb_ph))

				grads = tf.concat(grads, axis = 0)  
				# [num_classes, batchsize, maxlen, embedding_dims]

				approxs = grads * tf.expand_dims(emb_ph, 0) 
				# [num_classes, batchsize, maxlen, embedding_dims]
				self.sess = K.get_session()  
				self.grads = grads 
				self.approxs = approxs
				self.input_ph = X_ph
				self.emb_out = emb_out
				self.emb_ph = emb_ph
				weights_name = 'original.h5'
				model1.load_weights('{}/models/{}'.format(data, weights_name), 
					by_name=True)
				model2.load_weights('{}/models/{}'.format(data, weights_name), 
					by_name=True)  
				self.pred_model.load_weights('{}/models/{}'.format(data, weights_name), 
					by_name=True)
				print('Model constructed.', weights_name)
				# For validating the data. 
				emb_weights = emb_layer.get_weights() 
				emb_weights[0][0] = np.zeros(50)
				self.emb_weights = emb_weights[0]
				emb_layer.set_weights(emb_weights)
			else:
				pred_model = Model(X_ph, preds)
				pred_model.compile(
					loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy']) 
				self.pred_model = pred_model
				from load_data import Data
				dataset = Data(self.data, train = True)
				self.train(dataset) 
				print('Training is done.') 

		elif data == 'agccnn':
			from agccnn.data_helpers import create_vocab_set, construct_batch_generator, find_words_positions
			filter_kernels = [7, 7, 3, 3, 3, 3]
			dense_outputs = 1024
			self.charlen = 1014
			self.maxlen = 1014
			nb_filter = 256
			self.num_classes = 4
			self.vocab, self.reverse_vocab, self.vocab_size, self.vocab_check = create_vocab_set()
			self.embedding_dims = self.vocab_size
			self.type = 'char'
			K.set_learning_phase(1 if train else 0)
			#Define what the input shape looks like
			inputs = Input(shape=(self.charlen, self.vocab_size), name='input', dtype='float32')

			conv = Conv1D(filters = nb_filter, kernel_size= filter_kernels[0], padding = 'valid', activation = 'relu', input_shape=(self.charlen, self.vocab_size))(inputs)

			conv = MaxPooling1D(pool_size=3)(conv)

			conv1 = Conv1D(filters = nb_filter, kernel_size= filter_kernels[1], padding = 'valid', activation = 'relu')(conv)

			conv1 = MaxPooling1D(pool_size=3)(conv1) 

			conv2 = Conv1D(filters = nb_filter, kernel_size= filter_kernels[2], padding = 'valid', activation = 'relu')(conv1)
			conv3 = Conv1D(filters = nb_filter, kernel_size= filter_kernels[3], padding = 'valid', activation = 'relu')(conv2)
			conv4 = Conv1D(filters = nb_filter, kernel_size= filter_kernels[4], padding = 'valid', activation = 'relu')(conv3)
			conv5 = Conv1D(filters = nb_filter, kernel_size= filter_kernels[5], padding = 'valid', activation = 'relu')(conv4) 

			conv5 = MaxPooling1D(pool_size=3)(conv5)
			conv5 = Flatten()(conv5)

			#Two dense layers with dropout of .5
			z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
			z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))
			#Output dense layer with softmax activation
			pred = Dense(self.num_classes, activation='softmax', name='output')(z)
			grads = []
			for c in range(self.num_classes):
				grads.append(tf.gradients(pred[:,c], inputs))
			grads = tf.concat(grads, axis = 0)  
			# [num_classes, batchsize, self.charlen, embedding_dims]
			approxs = grads * tf.expand_dims(inputs, 0) 
			# [num_classes, batchsize, self.charlen, embedding_dims]
			model = Model(inputs, pred) 
			model.compile(
				loss='categorical_crossentropy', 
				optimizer="sgd",
				metrics=['accuracy']
			)  
			model.load_weights(
				'agccnn/params/crepe_model_weights-15.h5', 
				by_name=True
			)	

			self.sess = K.get_session()  
			self.grads = grads 
			self.approxs = approxs
			self.input_ph = inputs 
			self.model = model  
			
			from nltk.tokenize.moses import MosesDetokenizer
			from nltk import word_tokenize

			detokenizer = MosesDetokenizer()
			self.tokenize = word_tokenize
			self.detokenize = detokenizer.detokenize
			self.construct_batch_generator = construct_batch_generator
			self.find_words_positions = lambda sent: find_words_positions(
					sent, 
					word_tokenize(sent), 
					self.charlen, 
					self.vocab, 
					self.vocab_size, 
					self.vocab_check
				)
			self.find_chars_positions = lambda sent: find_words_positions(
					sent, 
					list(sent.lower().replace(' ', '')), 
					self.charlen, 
					self.vocab, 
					self.vocab_size, 
					self.vocab_check, 
					True
				)

		elif data == 'yahoolstm':
			self.maxlen = 400
			self.num_classes = 10
			self.num_words = 20000
			self.batch_size = 40 
			self.embedding_dims = 300
			if not train:
				K.set_learning_phase(0)

			X_ph = Input(shape=(self.maxlen,), dtype='int32') 
			emb_layer = Embedding(                
				input_dim=self.num_words + 1,
				output_dim= self.embedding_dims,  
				input_length=self.maxlen,
				name = "embedding",
				trainable=True)
			emb = emb_layer(X_ph)

			if train:
				preds = construct_original_network(emb, data)
			else:
				emb_ph = Input(shape=(self.maxlen,self.embedding_dims), dtype='float32')  
				preds = construct_original_network(emb_ph, data)

			if train:
				model = Model(X_ph, preds) 

				model.compile(
					loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy']
				)
			else:
				model1 = Model(X_ph, emb)
				model2 = Model(emb_ph, preds) 
				pred_out = model2(model1(X_ph)) 
				model = Model(X_ph, pred_out)
				model.compile(
					loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy']
				)
				# Construct gradients. 
				grads = []
				for c in range(self.num_classes):
					grads.append(tf.gradients(preds[:,c], emb_ph))

				grads = tf.concat(grads, axis = 0)  
				# [num_classes, batchsize, maxlen, embedding_dims]

				approxs = grads * tf.expand_dims(emb_ph, 0) 
				# [num_classes, batchsize, maxlen, embedding_dims]
				prev_epoch = 0; prev_itr = 7
				model1.load_weights(
					'yahoolstm/models/original-{}-{}.hdf5'.format(prev_epoch, prev_itr), 
					by_name = True
				)
				model2.load_weights(
					'yahoolstm/models/original-{}-{}.hdf5'.format(prev_epoch, prev_itr), 
					by_name = True
				)

				emb_weights = emb_layer.get_weights() 
				self.emb_weights = emb_weights
				self.emb_out = emb 
				self.emb_ph = emb_ph
				self.sess = K.get_session()  
				self.grads = grads 
				self.approxs = approxs
				self.input_ph = X_ph
			self.pred_model = model  
			self.type = 'word'
			if train:
				from load_data import Data
				print('Loading data...')
				dataset = Data(data, train = True)
				print('Training...')
				self.train(dataset)

	def train(self, dataset):
		if self.data == 'imdbcnn':
			epochs = 5
			batch_size = 40
			filepath = '{}/models/original.h5'.format(self.data)
			checkpoint = ModelCheckpoint(
				filepath, 
				monitor='val_acc',
				verbose=1, 
				save_best_only=True, 
				mode='max')
			callbacks_list = [checkpoint]
			self.pred_model.fit(dataset.x_train, dataset.y_train, validation_data=(dataset.x_val, dataset.y_val),callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)

		elif self.data == 'yahoolstm':
			model = self.pred_model
			if  'models' not in os.listdir(self.data):
				os.mkdir('{}/models'.format(self.data))  
			num_iters = int(math.ceil(len(dataset.x_train) * 1.0 / self.batch_size))
			num_val_iters = int(math.ceil(len(dataset.x_val) * 1.0 / self.batch_size))
			save_freq = 20
			save_interval = int(num_iters // save_freq)
			val_interval = 20
		
			np.random.seed(0)
			epochs = 3
			for e in range(epochs):
				print("epoch %d" % e)
				# random permutes the data.
				idx = np.random.permutation(len(dataset.x_train))
				x_train, y_train = dataset.x_train[idx], dataset.y_train[idx]

				val_batch_itr = 0 
				for i in range(0, num_iters):
					batch_x = x_train[i * self.batch_size: (i+1) * self.batch_size]
					batch_y = y_train[i * self.batch_size: (i+1) * self.batch_size]
					curr_loss, curr_acc = model.train_on_batch(batch_x, batch_y) 
					if i == 0:
						training_loss, training_acc = curr_loss, curr_acc
					else:
						training_loss = (i * training_loss + 1 * curr_loss) / float(i+1) 
						training_acc = (i * training_acc + 1 * curr_acc) / float(i+1) 
					if (i+1) % save_interval == 0:
						current_freq = (i+1) // save_interval
						model.save_weights('{}/models/original-{}-{}.hdf5'.format(self.data, e,current_freq))
						print('Model saved at Epoch {}, Step {}'.format(e, i))
					if (i+1) % val_interval == 0:
						current_itr = val_batch_itr % num_val_iters
						batch_x = dataset.x_val[current_itr * self.batch_size:(current_itr+1) * self.batch_size]
						batch_y = dataset.y_val[current_itr * self.batch_size:(current_itr+1) * self.batch_size]
						current_loss, current_acc = model.test_on_batch(batch_x, batch_y)
						
						if val_batch_itr == 0:
							val_loss, val_acc = current_loss, current_acc
						else: 
							val_loss = (val_batch_itr * val_loss + current_loss) / float(val_batch_itr+1)
							val_acc = (val_batch_itr * val_acc + current_acc) / float(val_batch_itr+1)

						val_batch_itr += 1

						print('Epoch: {} Step: {}; train_loss {}; train_acc {}; val_loss {}; val_acc {}'.format(e, i, training_loss, training_acc,val_loss, val_acc))

				model.save_weights('{}/models/original-{}.hdf5'.format(self.data, e))
				entire_val_loss, entire_val_acc = model.evaluate(dataset.x_val, dataset.y_val, verbose=0)
				print('Epoch: {}; loss {}; acc {}'.format(epoch, val_loss, val_acc))
				print('Epoch: {}; entire loss {}; acc {}'.format(epoch, entire_val_loss, entire_val_acc))
				print('Saving model at the end of the epoch...')

	def train_augment(self, dataset, new_data, method, changing_way):
		print('Training model on augmented data...')
		if self.data == 'imdbcnn':
			epochs = 8
			batch_size = 40
			filepath = '{}/models/augment_{}_{}.h5'.format(self.data, method, changing_way)
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
				verbose=1, save_best_only=True, mode='max')
			callbacks_list = [checkpoint]
			x = np.vstack([dataset.x_train, new_data[0]])
			y = np.vstack([dataset.y_train, new_data[1]])
			idx = np.random.permutation(len(x))
			x = np.array(x)[idx]; y = np.array(y)[idx]
			self.pred_model.fit(
				x, 
				y, 
				validation_data=(dataset.x_val, dataset.y_val), 
				callbacks = callbacks_list, 
				epochs=epochs, 
				batch_size=batch_size
			)

	def predict(self, x, verbose=0):
		if self.data in ['imdbcnn','yahoolstm']: 
			if type(x) == list or x.shape[1] < self.maxlen:
				x = np.array(sequence.pad_sequences(x, maxlen=self.maxlen)) 
			return self.pred_model.predict(x, batch_size = 2500, 
				verbose = verbose) 

		elif self.data == 'agccnn':
			# x should be a list of texts. 
			if isinstance(x[0], basestring):
				generator = self.construct_batch_generator(x, self.vocab, self.vocab_size, self.vocab_check, self.charlen, batchsize = 128)
				predictions = []
				for batch_data in generator:
					predictions.append(self.model.predict(batch_data, verbose = verbose)) 
				return np.concatenate(predictions, axis = 0)
			return self.model.predict(x, verbose = verbose)

	def compute_gradients(self, x):
		if self.data in ['imdbcnn','yahoolstm']:
			batchsize = 400
			num_iters = int(math.ceil(len(x) * 1.0 / batchsize))
			grads_val = []
			for i in range(num_iters): 
				batch_data = x[i * batchsize: (i+1) * batchsize] 
				batch_emb = self.sess.run(self.emb_out, 
						feed_dict = {self.input_ph: batch_data})  

				batch_grads = self.sess.run(self.grads, feed_dict = {self.emb_ph: batch_emb}) # [num_classes, batchsize, maxlen, embedding_dims]
				grads_val.append(batch_grads)

			grads_val = np.concatenate(grads_val, axis = 1)
			# [num_classes, num_data, maxlen, embedding_dims]

			pred_val = self.predict(x)
			# [num_data, maxlen, embedding_dims]
			gradients = grads_val[np.argmax(pred_val, axis = 1), range(len(pred_val))]
			return gradients #np.sum(abs(class_specific_scores), axis = -1)

		elif self.data == 'agccnn':
			generator = self.construct_batch_generator(x, self.vocab, self.vocab_size, self.vocab_check, self.charlen, batchsize = 128)
			grads_val = []
			for s, batch_data in enumerate(generator):
				grads_val.append(self.sess.run(self.grads, feed_dict = {self.input_ph: batch_data}))
			# [num_classes, num_data, charlen, embedding_dims]  
			grads_val = np.concatenate(grads_val, axis = 1)
			pred_val = self.predict(x)
			# [num_data, charlen, embedding_dims]
			class_specific_grads = grads_val[np.argmax(pred_val, axis = 1), range(len(pred_val))]
			return class_specific_grads


	def compute_taylor_approximation(self, x):
		if self.data in ['imdbcnn','yahoolstm']:
			batchsize = 128
			num_iters = int(math.ceil(len(x) * 1.0 / batchsize))
			approxs_val = []
			for i in range(num_iters): 
				batch_data = x[i * batchsize: (i+1) * batchsize] 
				batch_emb = self.sess.run(self.emb_out, 
						feed_dict = {self.input_ph: batch_data})  
				batch_approxs = self.sess.run(self.approxs, feed_dict = {self.emb_ph: batch_emb}) # [num_classes, batchsize, maxlen, embedding_dims]
				approxs_val.append(batch_approxs) 
			approxs_val = np.concatenate(approxs_val, axis = 1)
			# [num_classes, num_data, length, embedding_dims]
			pred_val = self.predict(x) 
			# [num_data, length, embedding_dims]
			class_specific_scores = approxs_val[np.argmax(pred_val, axis = 1), range(len(pred_val))] 
			# [num_data, length]
			return np.sum(class_specific_scores, axis = -1)
		elif self.data == 'agccnn':
			generator = self.construct_batch_generator(x, self.vocab, self.vocab_size, self.vocab_check, self.charlen, batchsize = 128)
			approxs_val = [] 
			indices = []
			for s, batch_data in enumerate(generator):
				approxs_val.append(self.sess.run(self.approxs, feed_dict = {self.input_ph: batch_data}))   
				for sent in x[128 * s: 128 * (s+1)]:   
					indices.append(self.find_words_positions(sent))
			# [num_classes, num_data, charlen, embedding_dims]  
			approxs_val = np.concatenate(approxs_val, axis = 1)
			# print(np.sum(approxs_val[0] != 0, axis = -1))
			pred_val = self.predict(x)
			# [num_data, charlen, embedding_dims]
			class_specific_approxs = approxs_val[np.argmax(pred_val, axis = 1), range(len(pred_val))]
			approxs_score = []
			for i, approxs_val in enumerate(class_specific_approxs):  
				approx_score = [np.sum(np.sum(approxs_val[start_idx:end_idx], axis = 0), axis = 0) for start_idx, end_idx in indices[i]] # [wordlen]
				approxs_score.append(np.array(approx_score))
				# print(np.array(approx_score).shape)
			return approxs_score

	def compute_integrated_gradients(self, x):
		if self.data in ['imdbcnn','yahoolstm']:
			batchsize = 20#128 if self.data == 'imdbcnn' else 40
			steps = 10
			approxs_val = []
			emb_vals = []
			num_iters1 = int(math.ceil(len(x) * 1.0 / batchsize))
			for i in range(num_iters1):  
				batch_data = x[i * batchsize: (i+1) * batchsize] 
				batch_emb = self.sess.run(self.emb_out, 
						feed_dict = {self.input_ph: batch_data}) 
				step_batch_emb = [batch_emb * float(s) / steps for s in range(1, steps+1)] 
				# [steps,batchsize, maxlen, embedding_dimension]
				emb_vals.append(step_batch_emb)
			emb_vals = np.concatenate(emb_vals, axis = 1)
			# [steps, num_data, maxlen, embedding_dimension] 
			emb_vals = np.reshape(emb_vals, [-1, self.maxlen, self.embedding_dims]) 
			num_iters = int(math.ceil(len(emb_vals) * 1.0 / batchsize))
			for i in range(num_iters):
				print(i) 
				batch_emb = emb_vals[i * batchsize: (i+1) * batchsize] 
				batch_approxs = self.sess.run(self.approxs, feed_dict = {self.emb_ph: batch_emb}) 
				# [num_classes, batchsize, maxlen, embedding_dims] 
				approxs_val.append(batch_approxs) 
			approxs_val = np.concatenate(approxs_val, axis = 1) 
			# [num_classes, steps * num_data, length, embedding_dims]
			approxs_val = np.reshape(approxs_val, 
				[self.num_classes, steps, len(x), self.maxlen, self.embedding_dims])
			approxs_val = np.mean(approxs_val, axis = 1) 
			pred_val = self.predict(x) 
			# [num_data, length, embedding_dims]
			class_specific_scores = approxs_val[np.argmax(pred_val, axis = 1), range(len(pred_val))]  
			# [num_data, length]
			return np.sum(class_specific_scores, axis = -1)

		elif self.data == 'agccnn':
			batchsize = 128
			generator = self.construct_batch_generator(x, self.vocab, self.vocab_size, self.vocab_check, self.charlen, batchsize = batchsize)
			steps = 100
			approxs_val = [] 
			indices = []
			for s, batch_data in enumerate(generator): 
				emb_vals = [batch_data * float(step) / steps for step in range(1, steps+1)]  
				batch_approxs = np.mean([self.sess.run(self.approxs, feed_dict = {self.input_ph: emb_val_s}) for emb_val_s in emb_vals], axis = 0)
				# [num_classes, batchsize, maxlen, embedding_dims]
				approxs_val.append(batch_approxs)
				for sent in x[batchsize * s: batchsize * (s+1)]:  
					indices.append(self.find_words_positions(sent))
			# [num_classes, num_data, charlen, embedding_dims]  
			approxs_val = np.concatenate(approxs_val, axis = 1) 
			pred_val = self.predict(x)
			# [num_data, charlen, embedding_dims]
			class_specific_approxs = approxs_val[np.argmax(pred_val, axis = 1), range(len(pred_val))] 
			approxs_score = []
			for i, approxs_val in enumerate(class_specific_approxs): 
				approx_score = [np.sum(np.sum(approxs_val[start_idx:end_idx], axis = 0), axis = 0) for start_idx, end_idx in indices[i]] # [wordlen]
				approxs_score.append(np.array(approx_score))
			return approxs_score
