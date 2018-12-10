import lstm_char_cnn
import preprocess as pr
import tensorflow as tf
import numpy as np
import os
import csv
from tqdm import tqdm


def make_voca():
	# make word2idx idx2word char2idx idx2char dictionary
	data_util.get_vocabulary(
			data_path+'ptb.train.txt', 
			top_voca=None, 
			char_voca=True, 
			save_path=data_savepath
		)

def make_data():
	train_set = data_util.make_model_dataset(
			data_path+'ptb.train.txt', 
			voca_path=data_savepath, 
			time_depth=time_depth, 
			word_length=word_length, 
			batch_size=batch_size
		)
	
	valid_set = data_util.make_model_dataset(
			data_path+'ptb.valid.txt', 
			voca_path=data_savepath, 
			time_depth=time_depth, 
			word_length=word_length, 
			batch_size=batch_size
		)
	
	test_set = data_util.make_model_dataset(
			data_path+'ptb.test.txt', 
			voca_path=data_savepath, 
			time_depth=time_depth, 
			word_length=word_length, 
			batch_size=batch_size
		)
	return train_set, valid_set, test_set


def load_voca():
	char2idx = data_util.load_data(data_savepath+'char2idx.npy', data_structure='dictionary')
	idx2char = data_util.load_data(data_savepath+'idx2char.npy', data_structure='dictionary')
	word2idx = data_util.load_data(data_savepath+'word2idx.npy', data_structure='dictionary')
	idx2word = data_util.load_data(data_savepath+'idx2word.npy', data_structure='dictionary')
	return char2idx, idx2char, word2idx, idx2word



data_path = './PTB_dataset/'
data_savepath = './npy/'
tensorflow_saver_path = './saver_large/'
tensorboard_path = './tensorboard_large/'

time_depth = 35
word_length = 65
batch_size = 20

#####
data_util = pr.preprocess()

# if firt start
make_voca()

# else
train_set, valid_set, test_set = make_data()
char2idx, idx2char, word2idx, idx2word = load_voca()
#####

# paper table2
cell_num = 650
voca_size = len(char2idx)
target_size = len(word2idx)
embedding_size = 15 # == projection size 
lstm_stack = 2 # L=2
highway_stack = 2
pad_idx = char2idx['</p>']
window_size = [1,2,3,4,5,6,7] 
filters = [min(i*50, 200) for i in window_size] 



def train(model, dataset, lr):
	loss = 0

	# for truncated bptt
	initial_state = sess.run(model.initial_state)
	for i in tqdm(range( len(dataset[0]) ), ncols=50):
		data = dataset[0][i] # [batch_size, time_depth, word_length]
		data = data.reshape(-1, word_length) # [batch_siz * time_depth, word_length]
		target = dataset[1][i] # [batch_size, time_depth]
		
		train_loss, _, initial_state = sess.run([model.cost, model.minimize, model.stacked_state_tuple],
				{
					model.data:data, 
					model.target:target, 
					model.lr:lr,
					model.keep_prob:0.5,
					model.initial_state:initial_state
				}
			)
		loss += train_loss
	
	loss /= len(dataset[0])
	perplexity = np.exp(loss)
	return loss, perplexity


def valid_or_test(model, dataset):
	loss = 0

	# for truncated bptt
	initial_state = sess.run(model.initial_state)
	for i in tqdm(range( len(dataset[0]) ), ncols=50):
		data = dataset[0][i] # [batch_size, time_depth, word_length]
		data = data.reshape(-1, word_length) # [batch_siz * time_depth, word_length]
		target = dataset[1][i] # [batch_size, time_depth]

		current_loss, initial_state = sess.run([model.cost, model.stacked_state_tuple],
				{
					model.data:data, 
					model.target:target, 
					model.keep_prob:1,
					model.initial_state:initial_state
				}
			)
		loss += current_loss
	
	loss /= len(dataset[0])
	perplexity = np.exp(loss)
	return loss, perplexity



def run(model, trainset, validset, testset, lr, restore=0):

	if not os.path.exists(tensorflow_saver_path):
		print("create save directory")
		os.makedirs(tensorflow_saver_path)
	

	with tf.name_scope("tensorboard"):
		train_ppl = tf.placeholder(tf.float32, name='train_loss')
		vali_ppl = tf.placeholder(tf.float32, name='vali_loss')
		test_ppl = tf.placeholder(tf.float32, name='test')

		train_summary = tf.summary.scalar("train_loss", train_ppl)
		vali_summary = tf.summary.scalar("vali_loss", vali_ppl)
		test_summary = tf.summary.scalar("test_accuracy", test_ppl)
				
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(tensorboard_path, sess.graph)


	prev_valid_perplexity = None
	for epoch in range(restore+1, 50+1):
		train_loss, train_perplexity = train(model, trainset, lr)
		valid_loss, valid_perplexity = valid_or_test(model, validset)
		test_loss, test_perplexity = valid_or_test(model, testset)

		print(train_loss, valid_loss, test_loss)
		print("epoch:", epoch, 'train_ppl:', train_perplexity, 'valid_ppl:', valid_perplexity, 'test_ppl:', test_perplexity, 'lr:', lr, '\n')
		
		# Optimization
		if (prev_valid_perplexity is not None) and ((prev_valid_perplexity - valid_perplexity) <= 1):
			lr /= 2

		prev_valid_perplexity = valid_perplexity

		if (epoch) % 5 == 0:
			model.saver.save(sess, tensorflow_saver_path+str(epoch)+".ckpt")
		
		# tensorboard
		summary = sess.run(merged, {
					train_ppl:train_perplexity, 
					vali_ppl:valid_perplexity,
					test_ppl:test_perplexity, 
				}
		)
		writer.add_summary(summary, epoch)


sess = tf.Session()

#with tf.variable_scope("lstm_char_cnn", initializer=tf.initializers.random_uniform(-0.05, 0.05)):
model = lstm_char_cnn.lstm_char_cnn(
		sess = sess,
		time_depth = time_depth,
		word_length = word_length,
		voca_size = voca_size,
		target_size = target_size,
		embedding_size = embedding_size,
		cell_num = cell_num,
		lstm_stack = lstm_stack,
		highway_stack = highway_stack,
		pad_idx = pad_idx,
		window_size = window_size, 
		filters = filters,
		batch_size = batch_size
	)

lr = 1.0
run(model, train_set, valid_set, test_set, lr, restore=0)
