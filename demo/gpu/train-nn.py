#!/usr/bin/env python3
import sys
sys.path.append('../..')
import tqdm
import pickle
import ase.io
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nnemb import GridDescriptor, EXTPOT, Grid, NeuralNetworkEmbeddingPotential
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')

extpot = EXTPOT('EXTPOT.final')
shape, elements = extpot.parse()
yfactor = -elements.min()
y = elements.reshape((-1,)) 

all_descriptors = None

#rcuts = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 12]
rcuts = [12]
filenames = ['rcut-{:.2f}.pckl'.format(rcut) for rcut in rcuts]
filenames.append('nn.pckl')
for filename in filenames:
    descriptor = pickle.load(open(filename, 'rb'))
    if all_descriptors is None: 
        all_descriptors = descriptor
    else:
        all_descriptors = np.hstack((all_descriptors, descriptor), )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('checking cuda availability: ', device)
sys.stdout.flush()

descriptor_dim = all_descriptors.shape[-1]
#fc_dims = [640, 320, 160, 80, 40]
#fc_dims = [160, 160, 80, 40, 20]
fc_dims = [160, 80, 40, 20]
nnemb = NeuralNetworkEmbeddingPotential(descriptor_dim, fc_dims)
nnemb.to(device)

X = all_descriptors
y /= yfactor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7018)
print(X_train.shape)
print(y_train.shape)



# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   

# Instantiate training and test data
batch_size = 256
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

#test_data = Data(X_test, y_test)
#test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)



loss_function = nn.MSELoss()
learning_rate = 1e-3
#optimizer = optim.LBFGS(nnemb.nn.parameters(), lr=learning_rate)
optimizer = optim.Adam(nnemb.parameters(), lr=learning_rate)

#num_epochs = 10000
#profiling
num_epochs = 10
loss_values = []

#print(next(nnemb.parameters()).device)
#sys.stdout.flush()
for epoch in tqdm.tqdm(range(num_epochs)):
#for epoch in range(num_epochs):
    #print(epoch)
    #sys.stdout.flush()
    for X, y in train_dataloader:
        X, y = X, y
        #print(X.is_cuda)
        #print(y.is_cuda)
        #sys.stdout.flush()
        optimizer.zero_grad()
        pred = nnemb(X)
        #print(pred.is_cuda)
        #sys.stdout.flush()
        loss = loss_function(pred, y)
        #loss_values.append(np.sqrt(loss.item()))
        loss.backward()
        optimizer.step()

num_loss_values = len(loss_values)
step = np.linspace(0, num_epochs, num_loss_values)
loss_values = np.array(loss_values)

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 14
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'Arial'

plt.plot(step, loss_values*yfactor)
plt.title('step-wise loss')
plt.xlabel('epochs')
plt.ylabel('RMSE loss')
plt.ylim([0, 1])
plt.savefig('training.png', dpi=600)
