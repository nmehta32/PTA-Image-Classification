
import numpy as np
import gzip
import random
import matplotlib.pyplot as plt
import struct

def Labels(name):
  f = gzip.open('/' + name)
  f.seek(4)
  images = struct.unpack('>I', f.read(4))[0]
  f.seek(8)
  labels = np.array(struct.unpack('>' + 'B' * images, f.read(images)))
  return (labels)

def Images(name):
  f = gzip.open('/'+name)
  f.seek(4)
  images = struct.unpack('>I', f.read(4))[0]
  c = struct.unpack('>I', f.read(4))[0]  
  r = struct.unpack('>I', f.read(4))[0]
  
  start = f.seek(16)
  m = r * c 
  pix_mat = []
  for i in range(images):
    f.seek(start + (i * m))
    pixel = np.array(struct.unpack('>' + 'B' * m, f.read(m)))
    pix_mat.append(pixel)
  return (np.array(pix_mat), images, r, c)
  
def test_train(ix,type):
  if (type == 'train'):
      numImages = TR_im
      labels  = TR_LB
  elif (type == 'test'):
      numImages = TS_IM
      labels = TS_LB
  if (ix <= numImages):
      return (labels[ix])

def Input(ix,type):
  d = np.zeros(10)
  label = test_train(ix,type)
  d[label] = 1
  return (d)

def PTA(W,n,eta, e):
  epoch = 0
  error = []
  pix_mat = trainingImgs
  labels = TR_LB
  while True:
      mis = 0
      for ix in range(n):
        v = W @ pix_mat[ix]
        if (np.argmax(v) != labels[ix]):
            mis += 1
      error.append(mis)
      epoch = epoch + 1
      for ix in range(n):
        W = W + eta*(Input(ix,type='train') - step_(W @ pix_mat[ix])).reshape(-1, 1) @ (
            pix_mat[ix].reshape(-1, 1).T)
      print(error[epoch - 1] / n)
      if ((error[epoch - 1] / n <= e ) or (epoch>15)):
        break
  return ( W ,error,epoch)

def testing( W, n):
  error = 0
  testImages =testImgs
  for ix in range(n):
      v = W @ testImages[ix]
      if (np.argmax(v) != TS_LB[ix]):
        error += 1
  return error

def step_(X):
  if (type(X) == np.ndarray or type(X) == list):
      for ix, x in enumerate(X):
        if (x >= 0):
            X[ix] = 1
        else:
            X[ix] = 0
  elif (type(X) == int):
        if (x >= 0):
            X = 1
        else:
            X = 0
  return (X)

def sign_(X):
  if (type(X) == np.ndarray or type(X) == list):
      for ix, x in enumeeta(X):
        if (x < 0):
            X[ix] = -1
        elif (x == 0):
            X[ix] = 0
        elif (x > 0 ):
            X[ix] = 1
  elif (type(X) == int):

    if (x < 0):
        X = -1
    elif (x == 0):
        X = 0
    elif (x > 0 ):
        X = 1
  return X

def plot(epoch,misList):
  plt.plot(np.array(range(epoch)),misList)
  plt.xlabel('epochs')
  plt.ylabel('no of misclassification')
  plt.show()

m = r * c
W = np.empty((0, m))
for j in range(10):
    w = np.array([random.uniform(-1, 1) for i in range(m)])
    W = np.vstack([w, W]) 
trainingImgs,TR_im,r,c = Images('train-images-idx3-ubyte')
TR_LB = Labels('train-labels-idx1-ubyte')
testImgs,TS_IM,_,_ = Images('t10k-images-idx3-ubyte')
TS_LB = Labels('t10k-labels-idx1-ubyte')
   

W_upd ,epoch_erros ,epoch = PTA(W,n=60000,eta=1,e=0.15)
plot(epoch,epoch_erros)
error = testing(W_upd, TS_IM)
print("testing error = ",error)
print("error % = ", error/100)
