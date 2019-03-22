# %%
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from pulseNet import pulseNet
import numpy as np

# %%
file_path = 'input/data.csv'
dataset = pd.read_csv(file_path,index_col=0)
dataset.describe()

# %%
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]
# Y = (Y==1)
# train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0)

# %%
def plot_graph(x_data, y_data, title):
    plt.plot(x_data, y_data)
    plt.xlabel('time')
    plt.ylabel('EEG')
    plt.title(title)
    plt.show()

def plot_ranodm(count=1):
    if(count<1):
        return
    row = random.randint(0,len(X))
    Xd = list(X.iloc[row])
    # Xd = [x1-x2 for (x1,x2) in zip(Xd[1:],Xd[:-1])]
    # Xd = [(int)(x/10) for x in Xd]
    # plot_graph(list(range(1,179)), list(X.iloc[row]), 'Plot for row {}: class {}'.format(row, Y[row]))
    plot_graph(list(range(1,179)), Xd, 'Plot for diff row {}: class {}'.format(row, Y[row]))
    plot_ranodm(count-1)
# X.iloc[random.randint(0,len(X))].plot.bar()

def plot_ranodm_image(X, count=1):
    if(count<1):
        return
    row = random.randint(0,len(X))
    Xd = X[row]
    model.plot_image(Xd)
    plot_ranodm_image(X, count-1)
plot_ranodm_image(5)

def percentAccuracy(val_predict, test_y):
    correct = [v==y for (v,y) in zip(val_predict,test_y)]
    print(sum(correct)/len(test_y)*100)

plot_ranodm(5)

# %%
model = pulseNet(pos_lim=150, neg_lim=-150, pulse_len=6)
X = X.values
Y = Y.values
Y = (Y==1)
# %%
model.transform_pulse(X)
matrix = np.asarray(model.pulses_to_matrix(), dtype=np.ndarray)
train_x, test_x, train_y, test_y = train_test_split(matrix, Y, random_state=0)

# %%
model.plot_image(matrix[0])