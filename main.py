from pulseNet import *
import numpy as np

model = pulseNet(pos_lim=2, neg_lim=-2, pulse_len=6)

sample_data = np.array([[1,2,3,4,5,4,3,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18],[-9,-5,0,0,2,3,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]])
Y = np.asarray([True,False])

model.transform_pulse(sample_data)

matrix = np.asarray(model.pulses_to_matrix(), dtype=np.ndarray)

# print(matrix[0])
model.export_jpgs(matrix, Y)