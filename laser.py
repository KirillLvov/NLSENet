import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

class NLSENet(nn.Module):
  def __init__(self, activation_func, der1_activation_func, der2_activation_func):
    super().__init__()
    self.layers = nn.ModuleList([nn.Linear(2, 100),
                                 nn.Linear(100, 100),
                                 nn.Linear(100, 100),
                                 nn.Linear(100, 100),
                                 nn.Linear(100, 2)])
    #for layer in self.layers:
    #  torch.nn.init.constant(layer.weight, 1)
    #  torch.nn.init.constant(layer.bias, 1)
    self.der_x1 = torch.Tensor([1,0])
    self.der_x1_temp = self.der_x1
    self.der_x1_x1 = torch.Tensor([0,0])
    self.der_x2 = torch.Tensor([0,1])
    self.der_x2_temp = self.der_x2
    self.der_x2_x2 = torch.Tensor([0,0])
    self.activation_func = activation_func
    self.der1_activation_func = der1_activation_func
    self.der2_activation_func = der2_activation_func

  def forward(self, z):
    self.der_x1 = torch.Tensor([1,0])
    self.der_x1_x1 = torch.Tensor([0,0])
    self.der_x2 = torch.Tensor([0,1])
    self.der_x2_x2 = torch.Tensor([0,0])
    for layer in self.layers:
      y = layer(z)
      z = self.activation_func(y)
      self.der_x1_temp = self.der1_activation_func(y) * (layer(self.der_x1) - layer.bias.data)
      self.der_x2_temp = self.der1_activation_func(y) * (layer(self.der_x2) - layer.bias.data)
      self.der_x1_x1 = self.der2_activation_func(y) * (layer(self.der_x1) - layer.bias.data)**2 + self.der1_activation_func(y) * (layer(self.der_x1_x1) - layer.bias.data)
      self.der_x2_x2 = self.der2_activation_func(y) * (layer(self.der_x2) - layer.bias.data)**2 + self.der1_activation_func(y) * (layer(self.der_x2_x2) - layer.bias.data)
      self.der_x1 = self.der_x1_temp
      self.der_x2 = self.der_x2_temp
    return y, self.der_x1, self.der_x2, self.der_x1_x1, self.der_x2_x2

def der1_tanh(x):
  return 1. - torch.tanh(x)**2

def der2_tanh(x):
  return -2. * torch.tanh(x) * (1. - torch.tanh(x)**2)

def der1_sigmoid(x):
  return torch.sigmoid(x) * (1. - torch.sigmoid(x))

def der2_sigmoid(x):
  return torch.sigmoid(x) * (1. - torch.sigmoid(x)) * (1. - 2. * torch.sigmoid(x))
  
def NLSELoss(E, E_t, E_xx):
  loss_re = -E_t[:,1] + 0.5 * E_xx[:,0] + torch.sum(E ** 2, dim = 1) * E[:,0]
  loss_im = E_t[:,0] + 0.5 * E_xx[:,1] + torch.sum(E ** 2, dim = 1) * E[:,1]
  loss = (loss_re ** 2 + loss_im ** 2).sum() / len(E)
  return loss
  
def InputField(x):
  ### dim(x) = [N, 2] = [N, (z,x)]
  ### coordinate = 0
  #phase = 10 * 3.14 * x[:,1]
  #return torch.hstack((torch.cos(phase).view(-1,1), torch.sin(phase).view(-1,1))) * torch.exp(-0.5 * (x[:,1] ** 2)).view(-1,1) # dim = [N, 2]
  return 2. * torch.hstack((torch.ones((x.size(0),1)), torch.zeros((x.size(0),1)))) / torch.cosh(x[:,0]).view(-1,1) # dim = [N, 2]

def InputValueLoss(x, E):
  ### dim(x) = [N, 2] = [N, (z,x)]
  ### dim(E) = [N, 2] = [N, (real, imag)]
  return ((InputField(x) - E) ** 2).sum() / len(E)

def InputValueLoss2(E_calc, E):
  ### dim(E_calc) = [N, 2] = [N, (real, imag)]
  ### dim(E) = [N, 2] = [N, (real, imag)]
  return ((E_calc - E) ** 2).sum() / len(E)

def BoundaryLoss(E):
  ### dim(E) = [N, 2] = [N, (real, imag)]
  return (E ** 2).sum() / len(E)
  
########################################
# Model create 
########################################

model = NLSENet(torch.tanh, der1_tanh, der2_tanh)
optimizer = torch.optim.Adam(model.parameters())

########################################
# Training only on initial condition
########################################

batch_size = 128
batch_input_size = 16
train_losses = []
test_losses = []

for epoch in range(1000):
  batch = torch.hstack((torch.rand((batch_size,1)) * 10 - 5, torch.rand((batch_size,1)) * np.pi * 0.5))
  model.train()
  output, der_x, der_t, der_xx, der_tt = model(batch)
  optimizer.zero_grad()
  loss = InputValueLoss2(InputField(batch), output)
  loss.backward()
  optimizer.step()
  model.train()
  train_losses.append(loss)
  testpred, der_x, der_t, der_xx, der_tt = model(testset)
  testloss = InputValueLoss2(InputField(testset), testpred)
  test_losses.append(testloss)
  print('Train loss: {:.4f}, Test loss: {:.4f}'.format(loss, testloss))

plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.yscale('log')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend()
plt.show()

#####################################
# Full training
#####################################

batch_input_size = 16
batch_boundary_size = 16
train_losses = []
test_losses = []
testset = torch.hstack((torch.rand((100,1)) * 10 - 5, torch.rand((100,1)) * np.pi * 0.5))
testset_input = torch.hstack((torch.rand((10,1)) * 10 - 5, torch.zeros((10,1))))
testset_boundary_left = torch.hstack((torch.ones((10,1)) * -5, torch.rand((10,1)) * np.pi * 0.5))
testset_boundary_right = torch.hstack((torch.ones((10,1)) * 5, torch.rand((10,1)) * np.pi * 0.5))

for epoch in range(1000):
  batch = torch.hstack((torch.rand((batch_size,1)) * 10 - 5, torch.rand((batch_size,1)) * np.pi * 0.5))
  batch_input = torch.hstack((torch.rand((batch_input_size,1)) * 10 - 5, torch.zeros((batch_input_size,1))))
  batch_boundary_left = torch.hstack((torch.ones((batch_boundary_size,1)) * -5, torch.rand((batch_boundary_size,1)) * np.pi * 0.5))
  batch_boundary_right = torch.hstack((torch.ones((batch_boundary_size,1)) * 5, torch.rand((batch_boundary_size,1)) * np.pi * 0.5))
  model.train()
  output, der_x, der_t, der_xx, der_tt = model(batch)
  output_input = model(batch_input)
  output_boundary_left = model(batch_boundary_left)
  output_boundary_right = model(batch_boundary_right)
  loss1 = NLSELoss(output, der_t, der_xx)
  loss2 = InputValueLoss(batch_input, output_input[0])
  loss3 = BoundaryLoss(output_boundary_left[0])
  loss4 = BoundaryLoss(output_boundary_right[0])
  loss = loss1 + loss2 + loss3 + loss4
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model.eval()
  train_losses.append(loss)
  testpred, der_x, der_t, der_xx, der_tt = model(testset)
  testpred_input = model(testset_input)
  testpred_boundary_left = model(testset_boundary_left)
  testpred_boundary_right = model(testset_boundary_right)
  testloss = NLSELoss(testpred, der_t, der_xx) + InputValueLoss(testset_input, testpred_input[0]) + BoundaryLoss(testpred_boundary_left[0]) + BoundaryLoss(testpred_boundary_right[0])
  test_losses.append(testloss)
  print('Train loss: {:.4f}, Test loss: {:.4f}'.format(loss, testloss))

plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.yscale('log')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend()
plt.show()
