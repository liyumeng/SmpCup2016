import os,sys
sys.path.append(os.path.abspath('.'))
print('正在进行年龄特征变换...')
import keras
from  keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten,Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling1D,Convolution1D,MaxPooling2D,AveragePooling2D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.engine.topology import Merge
from base.keras_helper import ModelCheckpointPlus
from base.yuml.models import StackEnsemble
from base.yuml.models import MCNN2
from base.dataset import load_v2,feature_path,smp_path
from base.utils import merge,describe,get,get_xs,get_word_vectors
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import SVC
from sklearn import cross_validation
from scipy import sparse
import numpy as np
import scipy.sparse as ss
from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold
ids,ys,f_train,f_test,f_content=load_v2()
f_text,f_stat,f_times,fp=f_train
f_text_test,f_stat_test,f_times_test,fp_test=f_test
ReTrain=False
y_age=ys[1]

import pickle
fids,f_w2v1,f_w2v1_test=pickle.load(open(feature_path+'/f_w2v_tfidf.300.cache','rb'))
fids,f_w2v2,f_w2v2_test=pickle.load(open(feature_path+'/f_word_svd.300.cache','rb'))

f_w2v=np.concatenate((f_w2v1,f_w2v2),axis=1)
f_w2v_test=np.concatenate((f_w2v1_test,f_w2v2_test),axis=1)

#--------------- MCNN ---------------
class MCNN(object):
    '''
    使用word2vec*tfidf的cnn并与人工特征混合，接口与sklearn分类器一致
    '''
    def __init__(self,cnn_input_dim,num_class=3):
        self.num_class=num_class
        self.build(cnn_input_dim)
        
    
    def build(self,vector_dim):
        #句子特征
        model=Sequential()
        model.add(Convolution2D(100,1,vector_dim,input_shape=(2,100,vector_dim),activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(50,1)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100,activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(3,activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'],)
        
        self.model=model
        self.earlyStopping=EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='auto')
        self.checkpoint=ModelCheckpointPlus(filepath='weights.hdf5',monitor='val_loss',verbose_show=20)
        
    def fit(self,X,y,Xvi=None,yvi=None):
        yc=to_categorical(y)
        if Xvi is None:
            self.model.fit(X,yc,nb_epoch=1000,verbose=0,validation_split=0.2,batch_size=32,callbacks=[self.earlyStopping,self.checkpoint])
        else:
            ycvi=to_categorical(yvi)
            self.model.fit(X,yc,nb_epoch=1000,verbose=0,validation_data=[Xvi,ycvi], 
                           batch_size=32,callbacks=[self.earlyStopping,self.checkpoint])
        self.model.load_weights('weights.hdf5')
        return self.model
    
    def predict(self,X):
        return predict_proba(X)
    
    def predict_proba(self,X):
        return self.model.predict(X)
    
#-----------------XGBoost --------------------------------

import xgboost as xgb
class XGB(object):
    def __init__(self):
        self.params={
                'booster':'gblinear',
                'eta':0.03,
                'alpha':0.1,
                'lambda':0,
                'subsample':1,
                'colsample_bytree':1,
                'num_class':3,
                'objective':'multi:softprob',
                'eval_metric':'mlogloss',
                'silent':1
            }
        pass
    
    def fit(self,X,y,Xvi=None,yvi=None):
        if Xvi is None:
            ti,vi=list(StratifiedShuffleSplit(y,test_size=0.2,random_state=100,n_iter=1))[0]
            dtrain=xgb.DMatrix(X[ti],label=y[ti])
            dvalid=xgb.DMatrix(X[vi],label=y[vi])
        else:
            dtrain=xgb.DMatrix(X,label=y)
            dvalid=xgb.DMatrix(Xvi,label=yvi)
        watchlist=[(dtrain,'train'),(dvalid,'val')]
        self.model=xgb.train(self.params,dtrain,num_boost_round=1000,early_stopping_rounds=25,evals=(watchlist),verbose_eval=100)
        return self.model
    
    def predict(self,X):
        return self.predict_proba(X)
    
    def predict_proba(self,X):
        return self.model.predict(xgb.DMatrix(X))

#----------- 获取特征 ------------
def get_xgb_X(f_train):
    f_text,f_stat,f_times,fp=f_train
    fs=(f_text[1],f_stat[:,[4,5,10,11,12,13,16]],)
    X=get_xs(fs)
    return X

#--------------------mcnn2-------------------------------------
def get_mcnn2_X(xtype='train'):
    if xtype=='train':
        f_text,f_stat,f_times,fp=f_train
        x_cnn=f_w2v
    else:
        f_text,f_stat,f_times,fp=f_test
        x_cnn=f_w2v_test
        
    f_norm_hour=f_times[:,38:62]
    f_hour=f_times[:,14:38]
    f_norm_week=f_times[:,7:14]
    f_week=f_times[:,0:7]
    fs=[f_text[12],f_text[3],f_stat[:,[4,5,10,11,12,13,16]],f_norm_hour]
    #fs=[f_text[12]]
    
    if xtype=='train':
        fs+=[em_xgb.get_next_input()[:3200]]
    else:
        fs+=[em_xgb.get_next_input()[3200:]]
    
    x_ext=get_xs(fs).toarray()
    return [x_cnn,x_ext]

if __name__=='__main__':
    filename=smp_path+'/models/yuml.age.feature'
    print('将输出文件：',filename)
    if ReTrain==True or os.path.exists(filename)==False:
        X=f_w2v
        X_test=f_w2v_test
        np.random.seed(100)
        
        #----mcnn model-----
        em_mcnn=StackEnsemble(lambda:MCNN(300),multi_input=False)
        f2_cnn=em_mcnn.fit(X,y_age)
        f2_cnn_test=em_mcnn.predict(X_test)
           
        #----xgb model-------
        X=get_xgb_X(f_train)
        X_test=get_xgb_X(f_test)
        em_xgb=StackEnsemble(lambda:XGB())
        f2_xgb=em_xgb.fit(X,y_age)
        f2_xgb_test=em_xgb.predict(X_test)
        
        #----mcnn2 model-----
        X=get_mcnn2_X('train')
        X_test=get_mcnn2_X('test')
        np.random.seed(100)

        em_mcnn2=StackEnsemble(lambda:MCNN2(X[0].shape[3],X[1].shape[1],num_channel=2),multi_input=True)
        f2_cnn=em_mcnn2.fit(X,y_age)
        f2_cnn_test=em_mcnn2.predict(X_test)

        #-----------------------特征输出-------------
        import pickle
        pickle.dump([em_mcnn2.get_next_input(),em_xgb.get_next_input(),em_mcnn.get_next_input()],open(filename,'wb'))
    print('程序运行完毕')
