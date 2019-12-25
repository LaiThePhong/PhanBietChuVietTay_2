# 1. Thêm các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

#Định nghĩa về hàm softmax
def softmax_stable(Z):
# Z = Z.reshape(Z.shape[0], -1)
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))# Tránh overflow
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

# Định nghĩa về hàm mất mát
def softmax_loss(X, y, W):# y là giá trị đầu ra thực one-hot (là list của chỉ số những giá trị có xác suất max), là matrix cỡ (Nx1), X matrix cỡ Nxd, W matrix cỡ dxC
    A = softmax_stable(X.dot(W)) # A là giá trị dự đoán, có cỡ (NxC)
    id0 = range(X.shape[0]) # indexes in axis 0, indexes in axis 1 are in y
    #⬆️ id0 chạy từ 0--> N-1
    return -np.mean(np.log(A[id0, y]))# Trả list(keepdims=False). Lấy giá trị trung bình của hàm mất mát của các điểm data 

# Định nghĩa hàm dự đoán. Từ các hàng (axis=1) matrix, chọn ra chỉ số của giá trị xác suất max trong hàng đó. 
def pred(W, X):# Trả list cỡ (Nx1)
    return np.argmax(X.dot(W), axis =1)#X cỡ (Nxd), W cỡ (dxC)

# Định nghĩa về Đạo hàm hàm mất mát
def softmax_grad(X, y, W):
    A = softmax_stable(X.dot(W)) # shape of (N, C)
    id0 = range(X.shape[0])
    A[id0,y]-=1 #A-Y,shapeof(N,C)#??????????????
    return X.T.dot(A)/X.shape[0]

# Định nghĩa hàm huấn luyện 
def softmax_fit(X, y, W, lr = 0.01, nepoches = 100, tol = 1e-5, batch_size = 10):
    W_old = W.copy()
    ep = 0
    loss_hist = [softmax_loss(X, y, W)] # store history of loss
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))# Làm cho số nbatches sang số nguyên
    while ep < nepoches:
        ep += 1
        mix_ids = np.random.permutation(N) # mix data. Xuất ra array data (List) sắp xếp ngẫu nhiên từ 0-> N 
        for i in range(nbatches):
            # get the i-th batch
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)]
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W -= lr*softmax_grad(X_batch, y_batch, W) # update gradient descent
        loss_hist.append(softmax_loss(X, y, W))
        if np.linalg.norm(W - W_old)/W.size < tol:
            break
        W_old = W.copy()
    return W, loss_hist

# 2. Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000] #shape of (10000,28,28)
X_train, y_train = X_train[:50000,:], y_train[:50000] # shape of (50000,28,28)
#print(X_train.shape)# Kq là shape=(50000,28,28) (Nxdxd)
#print(X_train.shape[2])
#print(y_train.shape)
#print(X_test.shape)

# Flatten matrix gốc theo các cột, xếp các cột thành hàng có 784 elements 
def Flatten(X):
    X_bar=np.zeros((X.shape[0],(X.shape[1]*X.shape[2]))) # Trả kq là matrix 0 có cỡ (50000x784)
    for i in range(X.shape[0]):
        X_creat = X[i].flatten('F')# Trả kq là array (list) cỡ 784x1
        X_bar[i]=X_creat # Gán kq sau khi flatten vào matrix mới  X_bar cỡ (50000,784)

    return X_bar # Trả kq là matrix cỡ (50000x784) là matrix X input dùng để tính sau này

# Reshape lại dữ liệu
X_bar_train = Flatten(X_train) # shape of (50000, 784)
X_bar_val = Flatten(X_val) # shape of (10000, 784)
X_bar_test = Flatten(X_test) # shape of (10000, 784)
#print(X_bar_train.shape)


# 3. One hot encoding label (Y)
Y_train = np_utils.to_categorical(y_train, 10) # shape of (50000,10)
Y_val = np_utils.to_categorical(y_val, 10) # shape of (10000,10)
Y_test = np_utils.to_categorical(y_test, 10) # shape of (10000,10)
#print('Dữ liệu y ban đầu ', Y_train[4])
#print('Shape of Y: ', Y_train.shape)
#print('Dữ liệu y sau one-hot encoding ',Y_train[3])
#plt.imshow(X_test[0].reshape(28,28), cmap='gray')
#plt.imshow(X_bar_test.[1].reshape(28,28).T, cmap='gray')
#plt.show()


#4. Training data
# Khởi tạo matrix trọng số ban đầu ngẫu nhiên cỡ (784x10)
def W(X,y):# y là giá trị thực
     W = np.random.rand(X.shape[1],y.shape[1])
     return W

#Chạy trên training set

W_0_train=W(X_bar_train,Y_train)# Matrix trọng số ban đầu

W_train,loss_hist = softmax_fit(X_bar_train,y_train,W_0_train,lr=0.01,nepoches=100,tol=1e-5,batch_size=10)
print(W_train)