import tensorflow as tf
import numpy as np

class lstm_char_cnn:
	def __init__(self, sess, time_depth, word_length, voca_size, target_size, embedding_size, cell_num, lstm_stack, highway_stack,
					pad_idx=0, window_size=[2,3,4], filters=[3,4,5]):
		self.sess = sess
		self.time_depth = time_depth
		self.word_length = word_length
		self.voca_size = voca_size # 'word': 단어 개수, 'char': char 개수
		self.target_size = target_size
		self.embedding_size = embedding_size # 512 == projection size 
		self.cell_num = cell_num # 4096
		self.lstm_stack = lstm_stack  
		self.highway_stack = highway_stack 
		self.pad_idx = pad_idx # 0
		self.window_size = window_size # for charCNN
		self.filters = filters # for charCNN  np.sum(filters) = 2048

		with tf.name_scope("placeholder"):
			self.data = tf.placeholder(tf.int32, [None, self.time_depth, self.word_length], name="char_x") # [N, time_depth, word_length]
			self.target = tf.placeholder(tf.int32, [None, self.time_depth], name="target") 
			self.lr = tf.placeholder(tf.float32, name="lr") # lr
			self.keep_prob = tf.placeholder(tf.float32, name="keep_prob") # keep_prob

		with tf.name_scope('mask'):
			self.target_pad_mask = tf.cast( #sequence_mask처럼 생성됨
						tf.not_equal(self.target, self.pad_idx),
						dtype=tf.float32
					) # [N, target_length] (include eos)

		with tf.name_scope("embedding"):		
			self.embedding_table = self.make_embadding_table(pad_idx=self.pad_idx)

			# charCNN
			self.before_embedding = self.charCNN(window_size=self.window_size, filters=self.filters) # [N, time_depth, sum(filters)]
			embedding = self.before_embedding

			# highway layer
			for i in range(self.highway_stack):
				embedding = self.highway_network(embedding=embedding, units=np.sum(filters)) # [N, time_depth, sum(filters)]
			self.after_embedding = embedding



		with tf.name_scope('prediction'):
			lstm_embedding = self.stacked_LSTM(self.after_embedding, self.lstm_stack) # [N, time_depth, sum(filters)]
			self.pred = tf.layers.dense(lstm_embedding, units=self.target_size, activation=None) # [N, time_depth, target_size]


		with tf.name_scope('train'): 
			target_one_hot = tf.one_hot(
						self.target, # [None, time_depth]
						depth=self.target_size,
						on_value = 1., # tf.float32
						off_value = 0., # tf.float32
					) # [N, time_depth, target_size]

			# calc cost
			self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, logits=self.pred) # [N, self.target_length]
			self.cost *= self.target_pad_mask # except pad
			self.cost = tf.reduce_sum(self.cost) / tf.reduce_sum(self.target_pad_mask) # == mean loss

			clip_norm = 5.0
			optimizer = tf.train.GradientDescentOptimizer(self.lr)
			grads_and_vars = optimizer.compute_gradients(self.cost)
			#https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
			clip_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm), gv[1]) for gv in grads_and_vars]
			self.minimize = optimizer.apply_gradients(clip_grads_and_vars)
			#self.minimize = optimizer.minimize(self.cost)


		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)


		self.sess.run(tf.global_variables_initializer())


	def make_embadding_table(self, pad_idx):
		zero = tf.zeros([1, self.embedding_size], dtype=tf.float32) # for padding
		embedding_table = tf.Variable(tf.random_uniform([self.voca_size-1, self.embedding_size], -0.05, 0.05)) 
		front, end = tf.split(embedding_table, [pad_idx, -1])
		embedding_table = tf.concat((front, zero, end), axis=0)
		return embedding_table

	def convolution(self, embedding, embedding_size, window_size, filters):
		convolved_features = []
		for i in range(len(window_size)):
			convolved = tf.layers.conv2d(
						inputs = embedding, 
						filters = filters[i], 
						kernel_size = [window_size[i], embedding_size], 
						strides=[1, 1], 
						padding='VALID', 
						activation=tf.nn.tanh
					) # [N, ?, 1, filters]
			convolved_features.append(convolved) # [N, ?, 1, filters] 이 len(window_size) 만큼 존재.
		return convolved_features


	def max_pooling(self, convolved_features):
		pooled_features = []
		for convolved in convolved_features: # [N, ?, 1, self.filters]
			max_pool = tf.reduce_max(
						input_tensor = convolved,
						axis = 1,
						keep_dims = True
					) # [N, 1, 1, self.filters]
			pooled_features.append(max_pool) # [N, 1, 1, self.filters] 이 len(window_size) 만큼 존재.
		return pooled_features


	def charCNN(self, window_size, filters):
		len_word = tf.shape(self.data)[1] # word length

		embedding = tf.nn.embedding_lookup(self.embedding_table, self.data) # [N, word, char, self.embedding_size] 
		embedding = tf.reshape(embedding, [-1, tf.shape(embedding)[2], self.embedding_size]) # [N*word, char, self.embedding_size]
			# => convolution 적용하기 위해서 word는 batch화 시킴. 동일하게 적용되도록.
		embedding = tf.expand_dims(embedding, axis=-1) # [N*word, char, self.embedding, 1]
			# => convolution을 위해 channel 추가.

		convolved_embedding = self.convolution(embedding, self.embedding_size, window_size, filters)
			# => [N*word, ?, 1, filters] 이 len(window_size) 만큼 존재.
		max_pooled_embedding = self.max_pooling(convolved_features=convolved_embedding)
			# => [N*word, 1, 1, filters] 이 len(window_size) 만큼 존재. 
		embedding = tf.concat(max_pooled_embedding, axis=-1) # [N*word, 1, 1, sum(filters)]
			# => filter 기준으로 concat
		embedding = tf.reshape(embedding, [-1, len_word, np.sum(filters)]) # [N, word, sum(filters)]
		return embedding		


	def highway_network(self, embedding, units):
		# embedding: [N, word, sum(filters)]
		transform_gate = tf.layers.dense(embedding, units=units, activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-2)) # [N, word, sum(filters)]
		carry_gate = 1-transform_gate # [N, word, sum(*filters)]
		block_state = tf.layers.dense(embedding, units=units, activation=tf.nn.relu)
		highway = transform_gate * block_state + carry_gate * embedding # [N, word, sum(filters)]
			# if transfor_gate is 1. then carry_gate is 0. so only use block_state
			# if transfor_gate is 0. then carry_gate is 1. so only use embedding
			# if transfor_gate is 0.@@. then carry_gate is 0.@@. so use sum of scaled block_state and embedding
		return highway



	def stacked_LSTM(self, data, stack):
		# data [N, self.time_depth, sum(self.filters)]

		fw_input = data
		for i in range(stack):
			cell = tf.contrib.rnn.LSTMCell(self.cell_num)
			# fw_input: shape: [N, self.time_depth, self.cell_num]
			fw_input, _ = tf.nn.dynamic_rnn(cell, fw_input, dtype=tf.float32, scope='stack_'+str(i))
			fw_input = tf.nn.dropout(fw_input, self.keep_prob)
		return fw_input


