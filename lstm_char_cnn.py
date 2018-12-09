import tensorflow as tf
import numpy as np

class lstm_char_cnn:
	def __init__(self, sess, time_depth, word_length, voca_size, target_size, embedding_size, cell_num, lstm_stack, highway_stack,
					pad_idx=0, window_size=[2,3,4], filters=[3,4,5], batch_size=20):
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
		self.batch_size = batch_size

		with tf.name_scope("placeholder"):
			# data: [N*time_depth, word_length], 이 형식으로 넣어줘야 나중에 char embedding만 추출 가능(data: [N, word_length])
			self.data = tf.placeholder(tf.int32, [None, self.word_length], name="char_x")
			self.target = tf.placeholder(tf.int32, [None, self.time_depth], name="target") 
			self.lr = tf.placeholder(tf.float32, name="lr") # lr
			self.keep_prob = tf.placeholder(tf.float32, name="keep_prob") # keep_prob

		with tf.name_scope("embedding"):		
			self.embedding_table = self.make_embadding_table(pad_idx=self.pad_idx)

			# charCNN
			self.before_embedding = self.charCNN(window_size=self.window_size, filters=self.filters) # [N*time_depth, sum(filters)]
			embedding = self.before_embedding # [N*time_depth, sum(filters)]

			# highway layer
			for i in range(self.highway_stack):
				embedding = self.highway_network(embedding=embedding, units=np.sum(filters)) # [N*time_depth, sum(filters)]
			self.after_embedding = embedding # [N*time_depth, sum(filters)]

		

		with tf.name_scope('prediction'):
			lstm_embedding, self.stacked_state_tuple = self.stacked_LSTM(self.after_embedding, self.lstm_stack) # [N, time_depth, self.cell_num]
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
			self.cost = tf.reduce_mean(self.cost)

			clip_norm = 5.0
			optimizer = tf.train.GradientDescentOptimizer(self.lr)
			grads_and_vars = optimizer.compute_gradients(self.cost)
			
			#https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
			clip_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm), gv[1]) for gv in grads_and_vars]
			self.minimize = optimizer.apply_gradients(clip_grads_and_vars)


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
		# embedding: [N*time_depth, word_length, self.embedding_size, 1]
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
			convolved_features.append(convolved) # [N*time_depth, ?, 1, filters] * len(window_size).
		return convolved_features


	def max_pooling(self, convolved_features):
		# convolved_features: [N*time_depth, ?, 1, filters] * len(window_size)
		pooled_features = []
		for convolved in convolved_features: # [N*time_depth, ?, 1, self.filters]
			max_pool = tf.reduce_max(
						input_tensor = convolved,
						axis = 1,
						keep_dims = True
					) # [N, 1, 1, self.filters]
			max_pool = tf.layers.flatten(max_pool) # [N*time_depth, self.filters[i]]
			pooled_features.append(max_pool) # [N*time_depth, self.filters[i]] * len(window_size).
		return pooled_features


	def charCNN(self, window_size, filters):
		len_word = tf.shape(self.data)[1] # word length

		embedding = tf.nn.embedding_lookup(self.embedding_table, self.data) # [N*time_depth, word_length, self.embedding_size] 
		embedding = tf.expand_dims(embedding, axis=-1) # [N*time_depth, word_length, self.embedding_size, 1]  => convolution을 위해 channel 추가.

		convolved_embedding = self.convolution(embedding, self.embedding_size, window_size, filters) # [N*time_depth, ?, 1, filters] * len(window_size).
		max_pooled_embedding = self.max_pooling(convolved_features=convolved_embedding) # [N*time_depth, self.filters[i]] * len(window_size).

		embedding = tf.concat(max_pooled_embedding, axis=-1) # [N*time_depth, sum(filters)], filter 기준으로 concat
		return embedding


	def highway_network(self, embedding, units):
		# embedding: [N*time_depth, sum(filters)]
		transform_gate = tf.layers.dense(embedding, units=units, activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-2)) # [N*time_depth, sum(filters)]
		carry_gate = 1-transform_gate # [N*time_depth, sum(filters)]
		block_state = tf.layers.dense(embedding, units=units, activation=tf.nn.relu) # [N*time_depth, sum(filters)]
		highway = transform_gate * block_state + carry_gate * embedding # [N*time_depth, sum(filters)]
			# if transfor_gate is 1. then carry_gate is 0. so only use block_state
			# if transfor_gate is 0. then carry_gate is 1. so only use embedding
			# if transfor_gate is 0.@@. then carry_gate is 0.@@. so use sum of scaled block_state and embedding
		return highway # [N*time_depth, sum(filters)]


	## truncated stacked LSTM  
	#https://www.tensorflow.org/tutorials/sequences/recurrent => truncated backprop
	def stacked_LSTM(self, data, stack):
		# data:  # [N*time_depth, sum(filters)]
		
		fw_input = tf.reshape(data, [-1, self.time_depth, np.sum(self.filters)]) # [N, time_depth, sum(filters)]

		cell_list = []
		for i in range(stack):
			cell = tf.contrib.rnn.LSTMCell(self.cell_num)
			cell_list.append(cell)

		initial_state_c = tf.zeros([stack*self.batch_size, self.cell_num]) # [stack*N, cell_num]
		initial_state_h = tf.zeros([stack*self.batch_size, self.cell_num]) # [stack*N, cell_num]
		
		# initial_state에 feed dict 해서 사용. (truncated bptt)
		self.initial_state = tf.contrib.rnn.LSTMStateTuple(c=initial_state_c, h=initial_state_h)


		split_initial_state_c = tf.split(self.initial_state.c, stack, axis=0) # stack 등분 
		split_initial_state_h = tf.split(self.initial_state.h, stack, axis=0) # stack 등분 

		fw_state_c_list = []
		fw_state_h_list = []
		for i in range(stack):
			c = split_initial_state_c[i]
			h = split_initial_state_h[i]
			init_state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)

			# fw_input shape: [N, self.time_depth, self.cell_num], fw_state c, h shape: [N, self.cell_num]
			fw_input, fw_state = tf.nn.dynamic_rnn(
					cell_list[i], 
					fw_input, 
					initial_state=init_state, 
					dtype=tf.float32, 
					scope='stack_'+str(i)
				)

			fw_input = tf.nn.dropout(fw_input, self.keep_prob)
			fw_state_c_list.append(fw_state.c)
			fw_state_h_list.append(fw_state.h)


		fw_state_c = tf.concat(fw_state_c_list, axis=0) # [stack*N, cell_num]
		fw_state_h = tf.concat(fw_state_h_list, axis=0) # [stack*N, cell_num]

		stacked_state_tuple = tf.contrib.rnn.LSTMStateTuple(c=fw_state_c, h=fw_state_h)
		return fw_input, stacked_state_tuple 