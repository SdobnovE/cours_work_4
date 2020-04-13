from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import copy
from scipy import stats
import random as rnd

class MGUA:
    def __init__(self, size_buf=10, 
                 num_selections = 10,
                 min_correlation = 0.95,
                model=LinearRegression()):
        '''
            size_buf - размер буфера
            num_selections - число селекций
            min_correlation - минимальная корреляция для добавления в буфер
        '''
        self.size_buf = size_buf
        self.num_selections = num_selections
        self.min_correlation = min_correlation
        self.model = model
        
        self.buf_r = np.array([0.0 for i in range(size_buf)])
        self.buf_size_ar = np.array([0 for i in range(num_selections)])

        self.buf_numbers = np.array([
            [-1 for i in range(size_buf)] 
            for k in range(num_selections)])
        
        self.buf_numbers_1 = np.array(
            [[-1 for i in range(size_buf)] 
             for k in range(num_selections)])

        
    
    
    def first_selection(self):
        iter_buf = 0
        m = len(self.data[0])
        
        n  = len(self.data)
        
        X1 = np.array([[float(i) for i in range(1)] for j in range(n)])
        
        for i in range(m):
            X1[:,0] = self.data[:,i]
            print(X1)
            print(self.y)
            self.model.fit(X1, self.y)
            if iter_buf < self.size_buf:
                tmp = self.model.predict(X1)
                ar = np.array([])

                for j in range (iter_buf):
                    t = abs(np.corrcoef (self.buf[0, :,j], tmp)[0,1])
                    ar = np.append (ar, t)
                
                if (iter_buf == 0) or (ar.max() < self.min_correlation):    
                    self.buf_r[iter_buf] = r2_score(tmp, self.y)
                    self.buf_numbers[0, iter_buf] = i
                    
                    self.buf[0, :, iter_buf] = tmp[:]
                    iter_buf += 1

            else:
                tmp = self.model.predict(X1)
                ind = self.buf_r.argmin()
                ar = np.array([])
                for j in range (self.buf.shape[2]):
                    t = abs(np.corrcoef (self.buf[0, :,j], tmp)[0,1])
                    ar = np.append (ar, t)
                
                if (
                    (ar.max() < self.min_correlation) 
                    and 
                    (self.buf_r[ind] < r2_score(tmp, self.y))
                   ):
                    
                    self.buf_r[ind] = r2_score(tmp, self.y)
                    self.buf_numbers[0, ind] = i
                    self.buf[0, :, ind] = tmp[:]
                    
        self.buf_numbers_1[0, :] = -2
        
        self.buf_size_ar[0] = iter_buf
        
    
    
    def __other_selections(self):
        m = len(self.data[0])
        n  = len(self.data)
        
        X2 = np.array([[float(i) for i in range(2)] for j in range(n)])
        for k in range (1, self.num_selections):
            iter_buf = 0
            for i in range(self.buf_size_ar[k - 1]):
                for j in range(0, m):  
                    #print(data[:,i].shape)
                    X2[:, 0] = self.buf[k - 1, :, i]
                    X2[:, 1] = self.data[:, j]
                    self.model.fit(X2, self.y)
                    if iter_buf < self.size_buf:
                        tmp = self.model.predict(X2)
                        ar = np.array([])

                        for j1 in range (iter_buf):
                            t = abs(np.corrcoef (self.buf[k,:,j1], tmp)[0,1])
                            ar = np.append (ar, t)
                        

                        if (iter_buf == 0) or (ar.max() < self.min_correlation):
                            
                            self.buf_r[iter_buf] = r2_score(tmp, self.y)
                            self.buf_numbers[k, iter_buf] = j
                            
                            self.buf_numbers_1[k, iter_buf] = i
                            self.buf[k, :,iter_buf] = tmp[:]
                            iter_buf += 1

                    else:
                        tmp = self.model.predict(X2)
                        ind = self.buf_r.argmin()
                        ar = np.array([])
                        for j1 in range (self.buf.shape[2]):
                            #print(self.buf.shape, i, j1)
                            t = abs(np.corrcoef (self.buf[k,:,j1], tmp)[0,1])
                            ar = np.append (ar, t)
                        if (
                            (ar.max() < self.min_correlation) 
                            and 
                            (self.buf_r[ind] < r2_score(tmp, self.y))
                        ):
                            self.buf_r[ind] = r2_score(tmp, self.y)
                            self.buf_numbers[k, ind] = j
                            self.buf_numbers_1[k, ind] = i
                            self.buf[k, :, ind] = tmp[:]
            self.buf_size_ar[k] = iter_buf

            print("selection", k + 1, "from", self.num_selections)
            
            
    def fit(self, X, y):
        n = len(y)
        self.buf = np.array(
            [[[float(i) for i in range(self.size_buf)] for j in range(n)] 
                for k in range(self.num_selections)])

        self.X1 = np.array([[float(i) for i in range(1)] for j in range(n)])
        self.X2 = np.array([[float(i) for i in range(2)] for j in range(n)])
        
        
        self.data = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        
        
        
        self.first_selection()
        self.__other_selections()
        
        lis = []
       
        
        for i in range(self.buf_size_ar[self.num_selections - 1]):
            lis.append([self.buf_numbers[self.num_selections - 1][i]])
        
        for i in range(self.num_selections - 1, 0, -1):
            for j in range(len(lis)):
                lis[j].append(self.buf_numbers[i - 1][self.buf_numbers_1[i,j]])
        
        for i in range(self.buf_size_ar[self.num_selections-1]):
            lis[i].sort()
        
        for i in range(len(lis)):
            lis[i] = np.unique(lis[i])
        self.lis = lis
        
    
    def predict_hist(self, X_test):
        '''принимает один вектор 
           рисует гистограмму всех ответов
        '''
        predic = []
        
        X_test = X_test.reshape((1,-1))
        for i in self.lis:
            self.model.fit(self.data[:,np.unique(i)], self.y)
            predic.append(self.model.predict(X_test[:,np.unique(i)]))
            
        predic = np.array(predic)
        predic -= predic.mean()
        predic /= predic.std()
        print(stats.kstest(predic, 'norm'))
        plt.hist(predic, bins = 8)
        
        
    def predict(self, X):
        predic = []
        for i in self.lis:
            self.model.fit(self.data[:,np.unique(i)], self.y)
            predic.append(self.model.predict(X[:,np.unique(i)]))
            
        predic = np.array(predic).T
        return predic.mean(axis=1)
        
    def score(self, X, y):
        res = self.predict(X)
        for i in self.lis:
            self.model.fit(self.data[:,np.unique(i)], self.y)
            print(self.model.score(X[:,np.unique(i)], y))
            
        return r2_score(res, y)
    def get_indexes(self):
        return self.lis