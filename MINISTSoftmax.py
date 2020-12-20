#引用所需的库并加载Mnist文件
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_folder):

  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
    paths.append(os.path.join(data_folder,fname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    
  return (x_train, y_train), (x_test, y_test)

#加载Mnist文件
(train_images, train_labels), (test_images, test_labels) = load_data('D:\MNIST\\')

X=train_images.reshape(60000,784).T
Y=np.eye(10)[train_labels].T

X_test=test_images.reshape(10000,784).T
Y_test=np.eye(10)[test_labels].T



print ("dimensions of X: " + str(X.shape))
print ("dimensions of Y: " + str(Y.shape))
print ("number of data: " + str(Y.shape[1]))





def sigmoid(z):
   
       return 1.0/(1+np.exp(-z))


def softmax(x):
    
        y = np.exp(x) / np.sum(np.exp(x), axis=0,keepdims=True)
        return y


""""inputs返回一个字典，其中包含所有w和b的值"""

def inputs( n_x , n_h ,n_y):
   


    W1 = np.random.normal(size=(n_h,n_x))*0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.normal(size=(n_y,n_h))*0.01
    b2 = np.zeros(shape=(n_y, 1))
    
  #par是一个参数库，储存所有w和b的值
    par = {"W1" : W1,
           "b1" : b1,
           "W2" : W2,
           "b2" : b2 }

    return par


'''foward函数进行前向传播，返回前向传播最终输出的Y值，和一个包含各层神经网络输入输出值的字典'''
def forward( X , par ):

    W1 = par["W1"]
    b1 = par["b1"]
    W2 = par["W2"]
    b2 = par["b2"]
    #前向传播计算Y1
    Z1 = np.dot(W1 , X) + b1
    H = sigmoid(Z1)
    Z2 = np.dot(W2 , H) + b2

    Y1 = softmax(Z2)

    store = {"Z1": Z1,
             "H": H,
             "Z2": Z2,
             "Y1": Y1}

    return (Y1, store)

'''total_cost是计算损失的函数，返回损失cost'''
def total_cost(Y1,Y):

    m = Y.shape[1]
    #计算成本
    cost = -1/m*np.sum(np.sum(Y*np.log(Y1),axis=0))

    return cost

'''backward函数用于反向传播，返回一个包含所有导数值的字典'''
def backward(par,store,X,Y):

  
    W2 = par["W2"]
    H = store["H"]
    Y1 = store["Y1"]

    dZ2=Y1-Y
    m = X.shape[1]
    dH = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dH,np.multiply(H,1-H) )
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    dW2 = (1 / m) * np.dot(dZ2, H.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2 }

    return grads

'''接收所有导数值，然后利用导数值更新参数，返回包含新参数的字典'''
def update(par,grad,lr=0.04):
    
    W1,W2 = par["W1"],par["W2"]
    b1,b2 = par["b1"],par["b2"]

    dW1,dW2 = grad["dW1"],grad["dW2"]
    db1,db2 = grad["db1"],grad["db2"]

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    par = {"W1": W1,
           "b1": b1,
           "W2": W2,
           "b2": b2}

    return par

"""sizes返回X和Y的形状"""

def sizes(X , Y):

    n_x = X.shape[0]
    n_y = Y.shape[0] 
    
    return (n_x,n_y)

'''循环'''
def train(X,Y,n_h,num,t=False):

    x_n = sizes(X, Y)[0]
    y_n = sizes(X, Y)[1]

    
    par = inputs(x_n,n_h,y_n) #储存w和b

    costs = []
    for i in range(num):
        Y1 , store = forward(X,par) 
        cost = total_cost(Y1,Y) #计算损失
        grad = backward(par,store,X,Y) #反向传播获得梯度
        par = update(par,grad,lr = 0.04) #储存更新后的w和b

        if t:
            if i%200 == 0:
                costs.append(cost)
                print("loop: ",i," cost: "+str(cost))
                
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(0.04))
    plt.show()
    return par


def predict(par,X,Y):

    Y1 , store = forward(X,par) #各项的值
    pred = np.argmax(Y1,axis=0).T
    
    print("accuracy:"  + str(float(np.sum(pred == np.argmax(Y,axis=0))/Y.shape[1])))
    return pred

par= train(X, Y, n_h = 10, num=2400, t=True)


predictions_test  = predict(par,X_test, Y_test) #测试集