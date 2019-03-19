import numpy as np

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
		side = self.categories ** self.pulse_len
		self.matrix = np.zeros(shape=(side, side))
		self.side = side


	def pulse_to_matrix(self):
		"""
		Return numpy 2d matrix with pulse plots
		"""
		self.init_mat()
		Y = self.Y
		for i in range(len(Y)):
			for j in range(len(Y[0])-self.pulse_len+1):
				side = self.side / 2
				row = 0
				col = 0
				for k in range(self.pulse_len):
					# start coord of ith row is j+k
					row += (Y[i][j+k] //2) * side
					col += (Y[i][j+k] % 2) * side
					side = side // self.categories
				self.matrix[row][col] += 1
		return self.matrix