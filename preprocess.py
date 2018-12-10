# https://nlp.stanford.edu/pubs/glove.pdf
import numpy as np
import collections
import os
import csv

class preprocess:
	def __init__(self):
		pass

	def get_vocabulary(self, data_path, top_voca=50000, char_voca=True, save_path=None):
		word_counter = collections.Counter({})
		if char_voca is True:
			char_counter = collections.Counter({})

		with open(data_path, 'r', newline='') as f:
			wr = csv.reader(f)
			for sentence in wr:
				sentence = sentence[0].split()
				word_counter += collections.Counter(sentence)

				if char_voca is True:
					for char in sentence:
						char_counter += collections.Counter(char)

		#빈도수 상위 top_voca개 뽑음. 튜플형태로 정렬되어있음 [("단어", 빈도수),("단어",빈도수)] 	
		word_counter = word_counter.most_common(top_voca) # => top_voca is None이면 전부 다.
		word2idx = {'</e>':0} # eos
		idx2word = {0:'</e>'} # eos

		for index, word in enumerate(word_counter):
			word2idx[word[0]] = index+1
			idx2word[index+1] = word[0]

		if char_voca is True:
			char_counter = char_counter.most_common(None)
			char2idx = {'</p>':0, '<unk>':1, '</g>':2, '</e>':3, '+':4} # pad, unk, go, eos
			idx2char = {0:'</p>', 1:'<unk>', 2:'</g>', 3:'</e>', 4:'+'} # pad, unk, go ,eos
			
			for index, char in enumerate(char_counter):
				char2idx[char[0]] = index+5
				idx2char[index+5] = char[0]

		
		if save_path is not None:
			if not os.path.exists(save_path):
				print("create save directory")
				os.makedirs(save_path)
			
			self.save_data(save_path+'word2idx.npy', word2idx)
			print("word2idx save", save_path+'word2idx.npy', len(word2idx))
			self.save_data(save_path+'idx2word.npy', idx2word)
			print("idx2word save", save_path+'idx2word.npy', len(idx2word))

			if char_voca is True:
				self.save_data(save_path+'char2idx.npy', char2idx)
				print("char2idx save", save_path+'char2idx.npy', len(char2idx))
				self.save_data(save_path+'idx2char.npy', idx2char)
				print("idx2char save", save_path+'idx2char.npy', len(idx2char))				
			
		
		if char_voca is True:
			return word2idx, idx2word, char2idx, idx2char	

		return word2idx, idx2word
		


	def make_model_dataset(self, data_path, voca_path=None, time_depth=35, word_length=65, batch_size=20):

		if os.path.exists(voca_path+'word2idx.npy') and os.path.exists(voca_path+'idx2word.npy') and os.path.exists(voca_path+'char2idx.npy') and os.path.exists(voca_path+'idx2char.npy'):
			char2idx = self.load_data(voca_path+'char2idx.npy', data_structure='dictionary')
			idx2char = self.load_data(voca_path+'idx2char.npy', data_structure='dictionary')
			word2idx = self.load_data(voca_path+'word2idx.npy', data_structure='dictionary')
			idx2word = self.load_data(voca_path+'idx2word.npy', data_structure='dictionary')				
		else:
			word2idx, idx2word, char2idx, idx2char = self.get_vocabulary(data_path, top_voca=None, char_voca=True, save_path=voca_path)
	
		with open(data_path, 'r', newline='') as f:
			wr = csv.reader(f)

			dataset_queue = []
			input_dataset = []
			target_dataset = []

			for sentence in wr: # sentence: [' consumers may want ~~ '] => sentence[0].split(): ['consumers', 'may', 'want', ~~ ]
				dataset_queue.extend(sentence[0].split() + ['</e>']) # append '</e>' to sentence and padding('</p>')

				while len(dataset_queue) > time_depth: # time_depth 이상이면 슬라이스해서 idx화 
					input_list = dataset_queue[:time_depth] # 입력부분 slice
					target_list = dataset_queue[1:1+time_depth] # 타겟부분 slice
					dataset_queue = dataset_queue[time_depth:] # dataset_queue 사용부분 dequeue
					
					# make input(char) idx
					input_list = self._word2charidx(
								word_list_1d=input_list, 
								char2idx_dict=char2idx, 
								word_length=word_length, 
								word_unk='<unk>',
								word_end='</e>',
								char_go='</g>', 
								char_end='</e>',
								char_pad='</p>'
							) # [time_depth, word_length]

					target_list = self._word2idx(
								word_list_1d=target_list, 
								word2idx_dict=word2idx, 
								word_unk='<unk>'
							) # [time_depth]

					input_dataset.append(input_list) # [-1, time_depth, word_length]
					target_dataset.append(target_list) # [-1, time_depth]

		input_dataset = np.array(input_dataset)[:(len(input_dataset)//batch_size) * batch_size] # batch size의 배수로 slice
		target_dataset = np.array(target_dataset)[:(len(input_dataset)//batch_size) * batch_size] # batch size의 배수로 slice
		
		input_dataset = input_dataset.reshape(batch_size, -1, time_depth, word_length)
		input_dataset = input_dataset.transpose(1, 0, 2, 3) # [-1, batch_size, time_depth, word_length]

		target_dataset = target_dataset.reshape(batch_size, -1, time_depth)
		target_dataset = target_dataset.transpose(1, 0, 2) # [-1, batch_size, time_depth]
		
		print(data_path, 'input_dataset', input_dataset.shape)
		print(data_path, 'target_dataset', target_dataset.shape)
		return input_dataset, target_dataset


	def _word2idx(self, word_list_1d, word2idx_dict, word_unk='<unk>'):
		word2idx_list = []
		for word in word_list_1d:
			if word in word2idx_dict:
				word2idx_list.append(word2idx_dict[word])
			else:
				word2idx_list.append(word2idx_dict[word_unk]) 
		return word2idx_list


	def _word2charidx(self, word_list_1d, char2idx_dict, word_length, word_unk='<unk>', word_end='</e>', 
				char_go='</g>', char_end='</e>', char_pad='</p>'):

		word2charidx_list = []
		for word in word_list_1d:
			if word == word_unk:
				word2char = ['</g>', '<unk>', '<unk>', '<unk>', '</e>'] # ['</g>', '<unk>', '</e>']
				#word2char = [char2idx_dict[char_go]] + ['<unk>'] + [char2idx_dict[char_end]] # ['</g>', '<unk>', '</e>']
				
			elif word == word_end:
				word2char = ['</g>', '+', '+', '+', '</e>'] # ['</g>', '+', '</e>']
				#word2char = [char2idx_dict[char_go]] + ['+'] + [char2idx_dict[char_end]] # ['</g>', '+', '</e>']
				
			else:
				word2char = [char2idx_dict[char_go]] + list(word) + [char2idx_dict[char_end]] # if word: 'my' => ['</g>', 'm', 'y', '</e>']
			
			char_list = self._word2idx(word2char, char2idx_dict, word_unk=word_unk)
			char_list = np.pad(char_list, (0, word_length-len(char_list)), 'constant', constant_values=char2idx_dict[char_pad])
			word2charidx_list.append(char_list)		
		return word2charidx_list



	def maximum_word(self, data_path):
		maximum = 0
		with open(data_path, 'r', newline='') as f:
			wr = csv.reader(f)

			for index, sentence in enumerate(wr):
				# append '</e>' to sentence and padding('</p>')
				sentence = sentence[0].split() + ['</e>']
				#print(index, len(sentence))
				maximum = max(maximum, len(sentence))
			print(data_path, maximum)



	def save_data(self, path, data):
		np.save(path, data)


	def load_data(self, path, data_structure = None):
		if data_structure == 'dictionary': 
			data = np.load(path, encoding='bytes').item()
		else:
			data = np.load(path, encoding='bytes')
		return data

	def read_csv_data(self, path):
		print('reading', path, end=' ')
		data = []
		with open(path, 'r', newline='') as o:
			wr = csv.reader(o)
			for i in wr:
				data.append(i)
		data = np.array(data)
		print(data.shape)
		return data

