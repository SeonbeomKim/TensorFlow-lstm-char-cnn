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
		word2idx = {'</p>':0, '</e>':1} # pad eos
		idx2word = {0:'</p>', 1:'</e>'} # pad eos

		for index, word in enumerate(word_counter):
			word2idx[word[0]] = index+2
			idx2word[index+2] = word[0]

		if char_voca is True:
			char_counter = char_counter.most_common(None)
			char2idx = {'</p>':0, '<unk>':1, '</g>':2, '</e>':3} # pad, unk, go, eos
			idx2char = {0:'</p>', 1:'<unk>', 2:'</g>', 3:'</e>'} # pad, unk, go ,eos
			
			for index, char in enumerate(char_counter):
				char2idx[char[0]] = index+4
				idx2char[index+4] = char[0]

		
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
		

	def make_char_idx_dataset_csv(self, data_path, except_word_dict, voca_path=None, save_path=None, time_depth=35, word_length=65):
		# except_word_dict = {</e>':0, '</p>':1, '<unk>':2, 'N':3} 
		if voca_path is not None:
			if os.path.exists(voca_path+'word2idx.npy') and os.path.exists(voca_path+'idx2word.npy') and os.path.exists(voca_path+'char2idx.npy') and os.path.exists(voca_path+'idx2char.npy'):
				char2idx = self.load_data(voca_path+'char2idx.npy', data_structure='dictionary')
				idx2char = self.load_data(voca_path+'idx2char.npy', data_structure='dictionary')
				word2idx = self.load_data(voca_path+'word2idx.npy', data_structure='dictionary')
				idx2word = self.load_data(voca_path+'idx2word.npy', data_structure='dictionary')				
			else:
				word2idx, idx2word, char2idx, idx2char = self.get_vocabulary(data_path, top_voca=None, char_voca=True, save_path=save_path)
		else:
			word2idx, idx2word, char2idx, idx2char = self.get_vocabulary(data_path, top_voca=None, char_voca=True, save_path=save_path)


		'''
		print('char2idx', len(char2idx))
		print('idx2char', len(idx2char))
		print('word2idx', len(word2idx))
		print('idx2word', len(idx2word))
		'''
		
		o = open(save_path, 'w', newline='')
		wt = csv.writer(o)

		with open(data_path, 'r', newline='') as f:
			wr = csv.reader(f)

			for sentence in wr:
				# append '</e>' to sentence and padding('</p>')
				sentence = sentence[0].split() + ['</e>']

				for start in range(len(sentence)):
					temp = sentence[start:start+time_depth+1]
					if len(temp) > time_depth:
						# make target(word) idx
						target_list = self._word2idx(
									word_list_1d=temp[1:], 
									word2idx_dict=word2idx, 
									unk='<unk>'
								) # [time_depth]
					
						# make input(char) idx
						input_list = self._word2charidx(
									word_list_1d=temp[:-1], 
									char2idx_dict=char2idx, 
									word_length=word_length, 
									except_word_dict=except_word_dict, 
									pad='</p>',
									unk='<unk>',
									go='</g>',
									end='</e>'
								) # [time_depth, word_length]
						input_list = np.reshape(input_list, [-1]) # input_list: [(time_depth)*word_length]
						input_target_concat = np.concatenate((input_list, target_list),axis=0)
						wt.writerow(input_target_concat)	
						break # 많은 데이터 뽑으려먼 break 제거.

					else:
						temp = np.pad(temp, (0, time_depth+1-len(temp)), 'constant', constant_values='</p>') # time_depth이 N이면 data는 N+1개 만들어야 함.
						# make target(word) idx
						target_list = self._word2idx(
									word_list_1d=temp[1:], 
									word2idx_dict=word2idx, 
									unk='<unk>'
								) # [time_depth]
					
						# make input(char) idx
						input_list = self._word2charidx(
									word_list_1d=temp[:-1], 
									char2idx_dict=char2idx, 
									word_length=word_length, 
									except_word_dict=except_word_dict, 
									pad='</p>',
									unk='<unk>',
									go='</g>',
									end='</e>'
								) # [time_depth, word_length]
						input_list = np.reshape(input_list, [-1]) # input_list: [(time_depth)*word_length]
						input_target_concat = np.concatenate((input_list, target_list),axis=0)
						wt.writerow(input_target_concat)
						# else문 실행되면 한번만 반영하고 종료.
						break

		o.close()
		print('ok', save_path)


	def _word2idx(self, word_list_1d, word2idx_dict, unk='<unk>'):
		word2idx_list = []
		for word in word_list_1d:
			if word in word2idx_dict:
				word2idx_list.append(word2idx_dict[word])
			else:
				word2idx_list.append(word2idx_dict[unk]) 
		return word2idx_list


	def _word2charidx(self, word_list_1d, char2idx_dict, word_length, except_word_dict, pad='</p>', unk='<unk>', go='</g>', end='</e>'):
		# except_word_dict: {'</e>':0} 이런식으로 pad char의 조합으로 표현해야하는 단어들의 딕셔너리.

		word2charidx_list = []
		for word in word_list_1d:
			if word in except_word_dict:
				char_list = [char2idx_dict[go], char2idx_dict[pad], char2idx_dict[end]] # ['</g>', '</p>', '</e>']
				char_list = np.pad(char_list, (0, word_length-len(char_list)), 'constant', constant_values=char2idx_dict[pad])
				word2charidx_list.append(char_list)
			else:
				word2char = [char2idx_dict[go]] + list(word) + [char2idx_dict[end]] # if word: 'my' => ['</g>', 'm', 'y', '</e>']
				char_list = self._word2idx(word2char, char2idx_dict, unk=unk)
				char_list = np.pad(char_list, (0, word_length-len(char_list)), 'constant', constant_values=char2idx_dict[pad])
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

