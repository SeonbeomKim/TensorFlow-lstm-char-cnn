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

def load_voca():
	char2idx = data_util.load_data(data_savepath+'char2idx.npy', data_structure='dictionary')
	idx2char = data_util.load_data(data_savepath+'idx2char.npy', data_structure='dictionary')
	word2idx = data_util.load_data(data_savepath+'word2idx.npy', data_structure='dictionary')
	idx2word = data_util.load_data(data_savepath+'idx2word.npy', data_structure='dictionary')
	return char2idx, idx2char, word2idx, idx2word


def _cosine_similarity(word1, word2):
	# word1: [N, sum(Filters)]
	# word2: [voca, sum(filters)]
	word1 = np.array(word1)
	word2 = np.array(word2)
	dot = np.dot(word1, word2.T) # [N, voca]
	word1_size = np.sqrt(np.sum(np.square(word1), axis=-1)) # [N]
	word2_size = np.sqrt(np.sum(np.square(word2), axis=-1)) # [voca]
	size = np.multiply(word1_size.reshape(-1, 1), word2_size) # [N, voca]
	cosim = dot/size # [N, voca]
	return cosim # [N, voca]


def _word2char_embedding(model, data_util, char2idx, word):
	# word embedding
	except_word_dict = {'</e>':0, '</p>':1, '<unk>':2} 
	word2char_list = data_util._word2charidx(
			word_list_1d=word, 
			char2idx_dict=char2idx, 
			word_length=word_length, 
			except_word_dict=except_word_dict, 
			pad='</p>', 
			unk='<unk>', 
			go='</g>', 
			end='</e>'
		) # [len(word), word_length]
	before_embedding, after_embedding = sess.run([model.before_embedding, model.after_embedding],
			{
				model.data:word2char_list
			}
		)
	return before_embedding, after_embedding


def _top_k_cosine_similarity(current_word_embedding, voca_embedding, embedding_name, word, top_k=None, name=None):
	cosim = _cosine_similarity(current_word_embedding, voca_embedding)
	argsort = np.argsort(-cosim)[:, :top_k] # decreasing order
	
	top_k_words = embedding_name[argsort] # [N, top_k]
	top_k_cosim = np.array([cosim[index][i] for index, i in enumerate(argsort)]) # [N, top_k]

	print(name)
	for i in range(len(word)):
		print('input_word:', word[i])
		print(np.array(list(zip(top_k_words[i], top_k_cosim[i]))))
		print()


def get_top_k_cosine_similarity(model, data_util, char2idx, word2idx, word, top_k=None):
	before_embedding = data_util.load_data(data_savepath+'before_embedding.npy')
	after_embedding = data_util.load_data(data_savepath+'after_embedding.npy')
	embedding_name = data_util.load_data(data_savepath+'word_embedding_name_order.npy')

	# word2char embedding
	current_word_before_embedding, current_word_after_embedding = _word2char_embedding(model, data_util, char2idx, word)
	_top_k_cosine_similarity(current_word_before_embedding, before_embedding, embedding_name, word, top_k=5, name='before_highway_cosine_similarity')
	_top_k_cosine_similarity(current_word_after_embedding, after_embedding, embedding_name, word, top_k=5, name='after_highway_cosine_similarity')



def make_npy_word2char_embedding(model, data_util, char2idx, word2idx):

	except_word_dict = {'</e>':0, '</p>':1, '<unk>':2} 
	embedding_name = list(word2idx.keys())
	word2char_list = data_util._word2charidx(
			word_list_1d=embedding_name, 
			char2idx_dict=char2idx, 
			word_length=word_length, 
			except_word_dict=except_word_dict, 
			pad='</p>', 
			unk='<unk>', 
			go='</g>', 
			end='</e>'
		) # [len(embedding_name), word_length]


	batch_size = 20
	before_embedding_list = []
	after_embedding_list = []
	for i in tqdm(range( int(np.ceil(len(word2char_list)/batch_size)) ), ncols=50):
		batch = word2char_list[batch_size * i: batch_size * (i + 1)] # [batch_size, word_length]
	
		before_embedding, after_embedding = sess.run([model.before_embedding, model.after_embedding],
				{
					model.data:batch
				}
			)
		before_embedding_list.append(before_embedding)
		after_embedding_list.append(after_embedding)

	before_embedding_list = np.concatenate(before_embedding_list, axis=0) #[len(word), sum(filers)]
	after_embedding_list = np.concatenate(after_embedding_list, axis=0) #[len(word), sum(filers)]

	data_util.save_data(data_savepath+"word_embedding_name_order.npy", np.array(embedding_name))
	print('ok', data_savepath+"word_embedding_name_order.npy")

	data_util.save_data(data_savepath+"before_embedding.npy", before_embedding_list)
	print('ok', data_savepath+"before_embedding.npy")

	data_util.save_data(data_savepath+"after_embedding.npy", after_embedding_list)
	print('ok', data_savepath+"after_embedding.npy")


char2idx, idx2char, word2idx, idx2word = load_voca()

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


sess = tf.Session()
with tf.variable_scope("lstm_char_cnn", initializer=tf.initializers.random_uniform(-0.05, 0.05)):
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

restore = 25
model.saver.restore(sess, tensorflow_saver_path+str(restore)+".ckpt")
print(tensorflow_saver_path+str(restore)+".ckpt")



# only first
#make_npy_word2char_embedding(model, data_util, char2idx, word2idx)


word = ['while', 'his', 'you', 'richard', 'trading', 'computer-aided', 'misinformed', 'looooook']
get_top_k_cosine_similarity(model, data_util, char2idx, word2idx, word, top_k=5)



#arithmetic test
before_king, after_king = _word2char_embedding(model, data_util, char2idx, ['king'])
before_man, after_man = _word2char_embedding(model, data_util, char2idx, ['man'])
before_woman, after_woman = _word2char_embedding(model, data_util, char2idx, ['woman'])
before_queen, after_queen = _word2char_embedding(model, data_util, char2idx, ['queen'])

before_calc = before_king - before_man + before_woman
after_calc = after_king - after_man + after_woman

before_embedding = data_util.load_data(data_savepath+'before_embedding.npy')
after_embedding = data_util.load_data(data_savepath+'after_embedding.npy')
embedding_name = data_util.load_data(data_savepath+'word_embedding_name_order.npy')

_top_k_cosine_similarity(before_calc, before_embedding, embedding_name, word=['king-man+woman'], top_k=5, name='before_king-man+woman')
_top_k_cosine_similarity(after_calc, after_embedding, embedding_name, word=['king-man+woman'], top_k=5, name='after_king-man+woman')

print('before (king-man+woman) vs (queen) cosine similarity')
print(_cosine_similarity(before_calc, before_queen), '\n')

print('after (king-man+woman) vs (queen) cosine similarity')
print(_cosine_similarity(after_calc, after_queen))