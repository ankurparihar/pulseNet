import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys
import os
from PIL import Image

class pulseNet():
	def __init__(self, pos_lim=150, neg_lim=-150, pulse_len=4):
		"""
		Parameters:
			pulse_len: moving buffer size
			matrix_out_size = output data size (square matrix)
			pulse_len is responsible for the out size and complexity
			matrix_out_size = category ** pulse_len
			######################################
			        Category Specification
			        ----------------------
			        0: + Under Limit
			        1: - Under Limit
			        2: +  Over Limit
			        3: -  Over Limit
			######################################
		"""
		# self.matrix = []
		self.pulse_len = pulse_len	# current implementation supports only 4
		# self.matrix_out_size = matrix_out_size
		self.categories = 4
		self.pos_lim = pos_lim
		self.neg_lim = neg_lim


	def transform_pulse(self, X):
		"""
		Parameters:
			X[rows x cols]: NumPy library defined 2d ndarray <class 'numpy.ndarray'>
		Return
			X[rows x (cols - 1)]: NumPy library defined 2d ndarray <class 'numpy.ndarray'>
		"""
		Y = []
		rows = len(X)
		cols = len(X[0])
		for i in range(rows):
			small_Y = []
			for j in range(cols-1):
				avg = (X[i][j] + X[i][j+1])/2
				if (avg > self.pos_lim) or (avg < self.neg_lim):
					over = 1
				else:
					over = 0
				if(X[i][j+1] >= X[i][j]):
					small_Y.append(over*2 + 0)
				else:
					small_Y.append(over*2 + 1)
			Y.append(small_Y)
		# return Y
		self.Y = Y


	def init_mat(self):
		"""
		Initialize matrix
		"""
		side = int(math.sqrt(self.categories) ** self.pulse_len)
		self.matrix = np.zeros(shape=(side, side), dtype=np.int32)
		self.side = side


	def pulses_to_matrix(self):
		"""
		Return list of numpy 2d matrix with pulse plots
		"""
		Y = self.Y
		self.final_mat = []
		divisor = math.sqrt(self.categories)
		for i in range(len(Y)):
			self.init_mat()
			for j in range(len(Y[0])-self.pulse_len+1):
				side = self.side / 2
				row = 0
				col = 0
				for k in range(self.pulse_len):
					# start coord of ith row is j+k
					row += (Y[i][j+k] //2) * side
					col += (Y[i][j+k] % 2) * side
					side = side // divisor
				self.matrix[int(row)][int(col)] += 1
			self.final_mat.append(self.matrix)
		return self.final_mat


	def plot_matrix(self,matrix):
		plt.imshow(np.asarray(matrix, dtype=float))
		plt.colorbar()
		plt.show()
	
	def plot_random_matrix(self, mat_list, y, count=1):
		if(count<1):
			return
		row = random.randint(0,len(mat_list)-1)
		mat = mat_list[row]
		self.plot_matrix(mat)
		print(row,y[row])
		self.plot_random_matrix(mat_list, y, count-1)

	def export_jpgs(self, matrix, y, out_dir='exports'):
		if not os.path.exists(out_dir):
  			os.mkdir(out_dir)
		categories = list(set(y))
		for cat in categories:
			cat_dir = out_dir + '/' + str(cat)
			if not os.path.exists(cat_dir):
				os.mkdir(cat_dir)
		for k in range(len(matrix)):
			k_dir = out_dir + '/' + str(y[k]) + '/' + str(k) + '.jpg'
			data = np.zeros((self.side, self.side, 3), dtype=np.uint32)
			for i in range(self.side):
				for j in range(self.side):
					color = matrix[k,i,j]
					data[i,j] = [color, color, color]
			img = Image.fromarray(data, 'RGB')
			img.save(k_dir, quality=100)
			if(k%500==0):
				print("Saving: {}".format(k+500))