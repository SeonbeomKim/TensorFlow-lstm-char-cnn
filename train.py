import lstm_char_cnn
import preprocess as pr
import tensorflow as tf
import numpy as np
import os
import csv
from tqdm import tqdm

data_util = pr.preprocess()

data_path = './PTB_dataset/'
data_savepath = './npy/'
tensorflow_saver_path = './saver/'
tensorboard_path = './tensorboard/'

time_depth = 35
word_length = 65


def make_voca_and_data():
	# make word2idx idx2word char2idx idx2char dictionary
	data_util.get_vocabulary(data_path+'ptb.train.txt', top_voca=None, char_voca=True, save_path=data_savepath)

	# make dataset csv  [N, time_depth * word_length + time_depth]
	except_word_dict = {'</e>':0, '</p>':1, '<unk>':2, 'N':3} 
	data_util.make_char_idx_dataset_csv(data_path+'ptb.train.txt', except_word_dict, voca_path=data_savepath, save_path=data_savepath+'train.csv', time_depth=time_depth, word_length=word_length)
	data_util.make_char_idx_dataset_csv(data_path+'ptb.valid.txt', except_word_dict, voca_path=data_savepath, save_path=data_savepath+'valid.csv', time_depth=time_depth, word_length=word_length)
	data_util.make_char_idx_dataset_csv(data_path+'ptb.test.txt', except_word_dict, voca_path=data_savepath, save_path=data_savepath+'test.csv', time_depth=time_depth, word_length=word_length)


def load_voca_and_data():
	char2idx = data_util.load_data(data_savepath+'char2idx.npy', data_structure='dictionary')
	idx2char = data_util.load_data(data_savepath+'idx2char.npy', data_structure='dictionary')
	word2idx = data_util.load_data(data_savepath+'word2idx.npy', data_structure='dictionary')
	idx2word = data_util.load_data(data_savepath+'idx2word.npy', data_structure='dictionary')

	# load dataset csv  [N, time_depth*word_length + time_depth]
	train_set = data_util.read_csv_data(data_savepath+'train.csv')
	valid_set = data_util.read_csv_data(data_savepath+'valid.csv')
	test_set = data_util.read_csv_data(data_savepath+'test.csv')
	
	return char2idx, idx2char, word2idx, idx2word, train_set, valid_set, test_set	



# if firt start
make_voca_and_data()


char2idx, idx2char, word2idx, idx2word, train_set, valid_set, test_set = load_voca_and_data()

# paper table2
cell_num = 300
voca_size = len(char2idx)
target_size = len(word2idx)
embedding_size = 15 # == projection size 
lstm_stack = 2 # L=2
highway_stack = 1
pad_idx = char2idx['</p>']
window_size = [1,2,3,4,5,6] 
filters = [i*25 for i in window_size] 

def train(model, dataset, lr):
	batch_size = 20
	loss = 0

	np.random.shuffle(dataset)

	for i in tqdm(range( int(np.ceil(len(dataset)/batch_size)) ), ncols=50):

		batch = dataset[batch_size * i: batch_size * (i + 1)] # [batch_size, 3]
		data = batch[:, :time_depth*word_length].reshape(-1, time_depth, word_length) # [batch_size, time_depth, word_length]
		target = batch[:, time_depth*word_length:] # [batch_size, time_depth] 

		train_loss, _ = sess.run([model.cost, model.minimize],
					{
						model.data:data, 
						model.target:target, 
						model.lr:lr,
						model.keep_prob:0.5
					}
				)
		loss += train_loss
	
	loss /= int(np.ceil(len(dataset)/batch_size))
	perplexity = np.exp(loss)
	return loss, perplexity


def valid_or_test(model, dataset):
	batch_size = 20
	loss = 0
	count = 0

	for i in tqdm(range( int(np.ceil(len(dataset)/batch_size)) ), ncols=50):
		count += 1

		batch = dataset[batch_size * i: batch_size * (i + 1)] # [batch_size, 3]
		data = batch[:, :time_depth*word_length].reshape(-1, time_depth, word_length) # [batch_size, time_depth, word_length]
		target = batch[:, time_depth*word_length:] # [batch_size, time_depth] 

		current_loss = sess.run(model.cost,
					{
						model.data:data, 
						model.target:target, 
						model.keep_prob:1.0
					}
				)
		loss += current_loss
	
	loss /= int(np.ceil(len(dataset)/batch_size))
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
	for epoch in range(restore+1, 20000+1):
		train_loss, train_perplexity = train(model, trainset, lr)
		valid_loss, valid_perplexity = valid_or_test(model, validset)
		test_loss, test_perplexity = valid_or_test(model, testset)

		print(train_loss, valid_loss, test_loss)
		print("epoch:", epoch, 'train_ppl:', train_perplexity, 'valid_ppl:', valid_perplexity, 'test_ppl:', test_perplexity, 'lr:', lr, '\n')
		
		# Optimization
		if (prev_valid_perplexity is not None) and ((prev_valid_perplexity - valid_perplexity) <= 1):
			lr /= 2

		prev_valid_perplexity = valid_perplexity

		if (epoch) % 10 == 0:
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
			filters = filters
		)

lr = 1.0
run(model, train_set, valid_set, test_set, lr, restore=0)
