import keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten,Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling1D,Convolution1D,MaxPooling2D,AveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.engine.topology import Merge
from base.keras_helper import ModelCheckpointPlus
from sklearn.metrics import log_loss,accuracy_score
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from collections import Counter
import pickle
import xgboost as xgb


class StackEnsemble(object):
    '''
    Stack可以利用一个元分类器，使用5折样本
    pred_proba: True调用predict_proba函数，False调用predict函数
    need_valid: True将给分类器的fit同时传入训练集和验证集（用于earlystop），False只传入训练集
    '''
    def __init__(self,model_creator,n_folds=5,seed=100,multi_input=False,pred_proba=True,need_valid=True):
        self.model_creator=model_creator
        self.n_folds=n_folds
        self.seed=seed
        self.multi_input=multi_input
        self.pred_proba=pred_proba
        self.need_valid=need_valid
        self.models=[]
        self.fit_yprob=[] #训练集的预测结果
        self.predict_yprob=[] #测试集的预测结果（5折bagging）
        self.indexes=None
        pass
    def fit(self,X,y):
        '''
        训练Stack Ensemble模型，会得到n_folds个分类器
        '''
        y_prob=np.zeros(1)
        indexes=StratifiedKFold(y,n_folds=self.n_folds,shuffle=True,random_state=self.seed)
        self.indexes=indexes
        cnt=0
        for ti,vi in indexes:
            print('--------stack-%d-------------'%(cnt+1))
            cnt+=1
            model=self.model_creator()
            #兼容keras的多输入
            if self.multi_input:
                Xti,Xvi=[x[ti] for x in X],[x[vi] for x in X]
            else:
                Xti,Xvi=X[ti],X[vi]
            
            #训练模型
            if self.need_valid:
                model.fit(Xti,y[ti],Xvi,y[vi])
            else:
                model.fit(Xti,y[ti])
                
            if self.pred_proba:
                y_p=model.predict_proba(Xvi)
            else:
                y_p=model.predict(Xvi)
            
            if y_prob.shape[0]==1:
                y_prob=np.zeros((len(y),y_p.shape[1]))
            y_prob[vi]=y_p
            
            self.models.append(model)
            print('log loss: %f, accuracy: %f'%(log_loss(y[vi],y_p),accuracy_score(y[vi],np.argmax(y_p,axis=1))))
        print('----------------------------------')
        print('----log loss: %f, accuracy: %f'%(log_loss(y,y_prob),accuracy_score(y,np.argmax(y_prob,axis=1))))
        
        self.fit_yprob=y_prob #得到对训练集的预测值
        return y_prob

    def predict(self,X,type='aver'):
        return self.predict_proba(X,type=type)
    
    '''
    type表示返回数据的格式
    aver: 取平均值返回
    raw: 返回原始数据（由多个分类器输出结果构成的数组）
    '''
    def predict_proba(self,X,type='aver'):
        if self.pred_proba:
            res= [model.predict_proba(X) for model in self.models]
        else:
            res= [model.predict(X) for model in self.models]
        self.predict_yprob=res
        
        if type=='aver':
            return np.average(res,axis=0)
        elif type=='max':
            return np.max(res,axis=0)
        else:
            return res
    '''
    获得下一级分类器的输入数据
    '''
    def get_next_input(self):
        aver=np.average(self.predict_yprob,axis=0)
        return np.vstack((self.fit_yprob,aver))
    
    '''
    保存模型
    '''
    def save(self,filename,save_model=False):
        if save_model==True:
            pickle.dump([self.models,self.fit_yprob,self.predict_yprob,self.indexes],open(filename,'wb'))
        else:
            pickle.dump([self.fit_yprob,self.predict_yprob,self.indexes],open(filename,'wb'))
        
    '''
    载入模型
    '''
    def load(self,filename):
        items=pickle.load(open(filename,'rb'))
        if len(items)==4:
            self.models,self.fit_yprob,self.predict_yprob,self.indexes=items
        else:
            self.fit_yprob,self.predict_yprob,self.indexes=items
          
    


class WeightEnsemble(object):
    '''
    对多个模型的输出结果进行加权，接口类似sklearn
    '''
    def __init__(self):
        pass
    
    def fit(self, y_probs, y_true):
        self.y_true=y_true
        self.y_probs=y_probs
        starting_values = [0.5]*len(y_probs)
        bounds = [(0,1)]*len(y_probs)
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        res=minimize(self.weight_log_loss, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
        self.weights=res.x
        return res
    
    def predict(self,y_probs):
        res=0
        for y_prob,w in zip(y_probs,self.weights):
            res+=y_prob*w
        return res
    
    def weight_log_loss(self,weights):
        final_ypred=0
        for weight,p in zip(weights,self.y_probs):
            final_ypred+=weight*p
        return log_loss(self.y_true,final_ypred)
    

class WeightVoter(object):
    '''
    获得获得不同stack的权重
    每个stack中的5个模型进行投票
    '''
    def __init__(self):
        self.models=[]
        
    def append(self,model):
        self.models.append(model)
        
    def extend(self,models):
        self.models.extend(models)
        
    def fit(self,y):
        self.y=y
        self.en_models=[]
        self.y_probs=[]
        index=0
        for ti,vi in self.models[0].indexes:
            fs=[em.fit_yprob[vi] for em in self.models]
            model=WeightEnsemble()
            model.fit(fs,self.y[vi])
            y_p=model.predict([em.predict_yprob[index] for em in self.models])
            self.y_probs.append(y_p)
            self.en_models.append(model)
            
            index+=1
        return self.en_models

    def vote(self):
        preds=np.argmax(self.y_probs,axis=2).transpose()
        y_pred=[sorted(Counter(items).items(),key=lambda x:x[1],reverse=True)[0][0] for items in preds]
        return y_pred
    
    def fit_vote(self,y):
        self.fit(y)
        return self.vote()
        
class MCNN2(object):
    '''
    cnn与人工特征混合，输入数据为2组
    使用word2vec*tfidf的cnn并与人工特征混合，接口与sklearn分类器一致
    '''
    def __init__(self,cnn_input_dim,ext_input_dim,num_class=3,num_channel=1,seed=100):
        self.seed=seed
        self.num_class=num_class
        self.num_channel=num_channel
        self.build(cnn_input_dim,ext_input_dim)
    
    def build(self,vector_dim,ext_feature_dim):
        #句子特征
        model=Sequential()
        model.add(Convolution2D(100,1,vector_dim,input_shape=(self.num_channel,100,vector_dim),activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(50,1)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100,activation='tanh'))
        model.add(Dropout(0.5))

        #用户整体特征
        model2=Sequential()
        model2.add(Dense(100,input_dim=ext_feature_dim,activation='tanh'))
        model2.add(Dropout(0.5))

        merged_model= Sequential()
        merged_model.add(Merge([model, model2], mode='concat', concat_axis=1))
        merged_model.add(Dense(self.num_class))
        merged_model.add(Activation('softmax'))

        merged_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'],)
        
        self.model=merged_model
        self.earlyStopping=EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='auto')
        self.checkpoint=ModelCheckpointPlus(filepath='weights.hdf5',monitor='val_loss',verbose_show=20)
        
    def fit(self,X,y,Xvi=None,yvi=None):
        np.random.seed(self.seed)
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
        return self.predict_proba(X)
    
    def predict_proba(self,X):
        return self.model.predict(X)
    

class MCNN3(object):
    '''
    cnn与人工特征混合，输入数据为3组
    使用word2vec*tfidf的cnn并与人工特征混合，接口与sklearn分类器一致
    '''
    def __init__(self,input_dims,num_class=8,num_channel=1,seed=100):
        self.seed=seed
        self.num_class=num_class
        self.num_channel=num_channel
        self.build(input_dims)
    
    def build(self,input_dims):
        #句子特征
        model=Sequential()
        model.add(Convolution2D(100,1,input_dims[0],input_shape=(self.num_channel,100,input_dims[0]),activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(50,1)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100,activation='tanh'))
        model.add(Dropout(0.5))

        #用户整体特征
        model2=Sequential()
        model2.add(Dense(100,input_dim=input_dims[1],activation='tanh'))
        model2.add(Dropout(0.5))
        
        #时间地域特征
        model3=Sequential()
        model3.add(Dense(output_dim=800,input_dim=input_dims[2],activation='tanh'))
        model3.add(Dropout(0.5))
        model3.add(Dense(output_dim=300,activation='tanh'))
        model3.add(Dropout(0.5))

        merged_model= Sequential()
        merged_model.add(Merge([model, model2,model3], mode='concat', concat_axis=1))
        merged_model.add(Dense(self.num_class))
        merged_model.add(Activation('softmax'))

        merged_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'],)
        
        self.model=merged_model
        self.earlyStopping=EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='auto')
        self.checkpoint=ModelCheckpointPlus(filepath='weights.hdf5',monitor='val_loss',verbose_show=20)
        
    def fit(self,X,y,Xvi=None,yvi=None):
        np.random.seed(self.seed)
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
        return self.predict_proba(X)
    
    def predict_proba(self,X):
        return self.model.predict(X)
    


class XGB(object):
    def __init__(self,params,early_stop=50,verbose=100):
        self.params=params
        self.early_stop=early_stop
        self.verbose=verbose
    
    def fit(self,X,y,Xvi=None,yvi=None):
        if Xvi is None:
            ti,vi=list(StratifiedShuffleSplit(y,test_size=0.2,random_state=100,n_iter=1))[0]
            dtrain=xgb.DMatrix(X[ti],label=y[ti])
            dvalid=xgb.DMatrix(X[vi],label=y[vi])
        else:
            dtrain=xgb.DMatrix(X,label=y)
            dvalid=xgb.DMatrix(Xvi,label=yvi)
        watchlist=[(dtrain,'train'),(dvalid,'val')]
        self.model=xgb.train(self.params,dtrain,num_boost_round=2000,early_stopping_rounds=self.early_stop,
                             evals=(watchlist),verbose_eval=self.verbose)
        return self.model
    
    def predict(self,X):
        return self.predict_proba(X)
    
    def predict_proba(self,X):
        return self.model.predict(xgb.DMatrix(X))