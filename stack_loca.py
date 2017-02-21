'''
本程序用来预测地域
运行前需要先运行process_data.py
将更新'/submission/temp.csv'文件中的地域一列
'''
import os,sys
sys.path.append(os.path.abspath('.'))
print('正在预测地域...')
from base.dataset import load_v2,feature_path,smp_path,load_v1
from base.yuml.models import StackEnsemble
from base.utils import describe,get,get_xs,get_word_vectors
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import cross_validation
from scipy import sparse
import numpy as np
import scipy.sparse as ss
from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold
import pickle

ReTrain=False

ids,ys,f_train,f_test,f_content=load_v2()
f_text,f_stat,f_times,fp=f_train
f_text_test,f_stat_test,f_times_test,fp_test=f_test
y_loca=ys[2]

fids,fp_sloca,fp_sprov,fp_sexist=pickle.load(open(feature_path+'/loca.source.feature','rb'))
n_folds=5

#--------------------------定义神经网络模型----------------------------------------------
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import SGD
from base.keras_helper import ModelCheckpointPlus
class LocaNN(object):
    '''
    3层BP神经网络
    '''
    def __init__(self,input_dim,seed=100):
        self.seed=seed
        self.build(input_dim)
    
    def build(self,input_dim):
        model=Sequential()
        model.add(Dense(output_dim=800,input_dim=input_dim,activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim=300,activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim=8))
        model.add(Dropout(0.3))
        model.add(Activation('softmax'))

        model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy'])

        self.model=model
        self.earlyStopping=EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
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
        return self.predict_proba(X)
    
    def predict_proba(self,X):
        return self.model.predict(X)
    
def get_nn_x(ftype):
    if ftype=='train':
        f_text,f_stat,f_times,fp=f_train
        ft_norm_hour=f_times[:,38:62]
        ft_norm_cnt=f_times[:,230:398]
        fs=(fp[0],fp[1],fp[2],fp[3],fp[4],ft_norm_hour,ft_norm_cnt,)
    else:
        f_text,f_stat,f_times,fp=f_test
        ft_norm_hour=f_times[:,38:62]
        ft_norm_cnt=f_times[:,230:398]
        fs=(fp[0],fp[1],fp[2],fp[3],fp[4],ft_norm_hour,ft_norm_cnt,)
    return get_xs(fs)

    
def get_nn_x2(ftype):
    if ftype=='train':
        f_text,f_stat,f_times,fp=f_train
        ft_norm_hour=f_times[:,38:62]
        ft_norm_cnt=f_times[:,230:398]
        fs=(fp[0],fp[1],fp[2],fp[3],fp[4],ft_norm_hour,ft_norm_cnt,
            fp_sloca[:3200],fp_sprov[:3200],fp_sexist[:3200],
            em_age.get_next_input()[:3200],
           em_gen.get_next_input()[:3200])
    else:
        f_text,f_stat,f_times,fp=f_test
        ft_norm_hour=f_times[:,38:62]
        ft_norm_cnt=f_times[:,230:398]
        fs=(fp[0],fp[1],fp[2],fp[3],fp[4],ft_norm_hour,ft_norm_cnt,
            fp_sloca[3200:],fp_sprov[3200:],fp_sexist[3200:],
            em_age.get_next_input()[3200:],
           em_gen.get_next_input()[3200:])
    return get_xs(fs)

#----------------填充空值----------------------------------------------
#填充y的空值
loca_empth_path=feature_path+'/loca.empty.pkl'
if os.path.exists(loca_empth_path):
    loca_empty=pickle.load(open(loca_empth_path,'rb'))
    tindexs=np.arange(len(ys[2]))[ys[2]==-1]
    if len(tindexs)>0:
        assert((tindexs==loca_empty[0]).all())
        y_loca[tindexs]=loca_empty[1]
else:
    filter_index=np.arange(len(ys[2]))[ys[2]!=-1]
    print('正在填充训练集中空缺的y值...')
    X=get_nn_x('train').toarray()
    X_test=get_nn_x('test').toarray()
    em_nn=StackEnsemble(lambda:LocaNN(X.shape[1]),n_folds=5)
    em_nn.fit(X[filter_index],y_loca[filter_index])
    
    tindexs=np.arange(len(ys[2]))[ys[2]==-1]
    loca_empty=np.argmax(em_nn.predict(X[tindexs]),axis=1)
    pickle.dump([tindexs,loca_empty],open(loca_empth_path,'wb'))
    y_loca[tindexs]=loca_empty[1]
    print('y值填充完毕')
    
#----------------1. BP神经网络模型--------------------------------


X=get_nn_x('train').toarray()
X_test=get_nn_x('test').toarray()
em_nn=StackEnsemble(lambda:LocaNN(X.shape[1]),n_folds=n_folds)

em_nn_path=smp_path+'/models/loca.em_nn.weight'
if ReTrain==False and os.path.exists(em_nn_path):
    em_nn.load(em_nn_path)
else:
    np.random.seed(50)
    em_nn.fit(X,y_loca)
    em_nn.predict(X_test)
    f2_nn=em_nn.get_next_input()
    em_nn.save(em_nn_path)
    
#-----------------2. KNN --------------------------------------------
from sklearn.neighbors import KNeighborsClassifier as KNN
import scipy.sparse as ss

def get_knn_x(xtype):
    if xtype=='train':
        f_text,f_stat,f_times,fp=f_train
        ft_norm_hour=f_times[:,38:62]
        ft_norm_cnt=f_times[:,230:398]
        fs=(fp[0],fp[1],fp[3],fp[2],fp[4],ft_norm_hour,ft_norm_cnt,)
       
    else:
        f_text,f_stat,f_times,fp=f_test
        ft_norm_hour=f_times[:,38:62]
        ft_norm_cnt=f_times[:,230:398]
        fs=(fp[0],fp[1],fp[3],fp[2],fp[4],ft_norm_hour,ft_norm_cnt, )
       
    return get_xs(fs)
X=get_knn_x('train')
X_test=get_knn_x('test')

em_knn_path=smp_path+'/models/loca.em_knn.weight'
em_knn=StackEnsemble(lambda:KNN(n_neighbors=20),need_valid=False,n_folds=n_folds)

if ReTrain==False and os.path.exists(em_knn_path):
    em_knn.load(em_knn_path)
else:
    em_knn.fit(X,y_loca)
    em_knn.predict(X_test)
    f2_knn=em_knn.get_next_input()
    em_knn.save(em_knn_path)

#-----------------3. MCNN -----------------------------------------------------
import pickle
fids,f_w2v1,f_w2v1_test=pickle.load(open(feature_path+'/f_w2v_tfidf.300.cache','rb'))
fids,f_w2v2,f_w2v2_test=pickle.load(open(feature_path+'/f_word_svd.300.cache','rb'))
#fids,f_w2v3,f_w2v3_test=pickle.load(open(feature_path+'/f_source_svd.300.cache','rb'))
#fids,f_w2v4,f_w2v4_test=pickle.load(open(feature_path+'/f_letter_svd.300.cache','rb'))
fids,f_sens=pickle.load(open(feature_path+'/f_sens.300.pkl','rb'))
f_sens=f_sens.reshape((f_sens.shape[0],1,100,300))
f_w2v=np.concatenate((f_w2v1,f_w2v2,f_sens[:3200]),axis=1)
f_w2v_test=np.concatenate((f_w2v1_test,f_w2v2_test,f_sens[3200:]),axis=1)


from base.yuml.models import MCNN2
def get_mcnn_x(xtype='train'):
    if xtype=='train':
        f_text,f_stat,f_times,fp=f_train
        x_cnn=f_w2v
        x_ext=f_text[2]
    else:
        f_text,f_stat,f_times,fp=f_test
        x_cnn=f_w2v_test
        x_ext=f_text[2]
    return [x_cnn,x_ext]

X=get_mcnn_x('train')
X_test=get_mcnn_x('test')

em_mcnn_path=smp_path+'/models/loca.em_mcnn.weight'
em_mcnn=StackEnsemble(lambda:MCNN2(X[0].shape[3],X[1].shape[1],num_class=8,num_channel=3),multi_input=True,n_folds=n_folds)
if ReTrain==False and os.path.exists(em_mcnn_path):
    em_mcnn.load(em_mcnn_path)
else:
    em_mcnn.fit(X,y_loca) 
    em_mcnn.predict(X_test)
    f2_mcnn=em_mcnn.get_next_input()
    em_mcnn.save(em_mcnn_path)
    
#-------------------4. MCNN3-----------------------------------------------------
f_w2v=np.concatenate((f_w2v1,f_w2v2,f_sens[:3200]),axis=1)
f_w2v_test=np.concatenate((f_w2v1_test,f_w2v2_test,f_sens[3200:]),axis=1)

from base.yuml.models import MCNN3
def get_mcnn3_x(xtype='train'):
    if xtype=='train':
        f_text,f_stat,f_times,fp=f_train
        x_cnn=f_w2v
    else:
        f_text,f_stat,f_times,fp=f_test
        x_cnn=f_w2v_test
        
    x_ext=f_text[2]
    
    ft_norm_hour=f_times[:,38:62]
    ft_norm_cnt=f_times[:,230:398]
    if xtype=='train':
        fs=(fp[0],fp[1],fp[2],fp[3],fp[4],ft_norm_hour,ft_norm_cnt)
    else:
        fs=(fp[0],fp[1],fp[2],fp[3],fp[4],ft_norm_hour,ft_norm_cnt)
    x_ext1=get_xs(fs).toarray()
    
    return [x_cnn,x_ext,x_ext1]

X=get_mcnn3_x('train')
X_test=get_mcnn3_x('test')

shape=(X[0].shape[3],X[1].shape[1],X[2].shape[1])

em_mcnn3_path=smp_path+'/models/loca.em_mcnn3.weight'
em_mcnn3=StackEnsemble(lambda:MCNN3(shape,num_class=8,num_channel=3),multi_input=True,n_folds=n_folds)

if ReTrain==False and os.path.exists(em_mcnn3_path):
    em_mcnn3.load(em_mcnn3_path)
else:
    em_mcnn3.fit(X,y_loca)
    em_mcnn3.predict(X_test)
    f2_mcnn3=em_mcnn3.get_next_input()
    em_mcnn3.save(em_mcnn3_path)
    

#------------------------ 6. 加权投票--------------------------------------------------------
from base.yuml.models import WeightVoter
ems=[em_nn,em_mcnn,em_knn,em_mcnn3]
voter=WeightVoter()
voter.extend(ems)
y_pred=voter.fit_vote(y_loca)

#-------------------------输出---------------------------------------------------------
from base.dataset import submission_path

loca_enum='华北,华东,华南,西南,华中,东北,西北,境外'.split(',')
y_pred=[loca_enum[y] for y in y_pred]

#-------------- output -----------------------
y_dict={}
for id,y in zip(ids[1],y_pred):
    y_dict[id]=y

with open(submission_path+'/empty.csv',encoding='utf8') as f:
    items=[item.strip() for item in f]

with open(submission_path+'/temp.csv','w',encoding='utf8') as f:
    f.write('%s\n'%items[0])
    cnt=0
    for item in items[1:]:
        values=item.split(',')
        if y_dict[values[0]]!=values[3]:
            #print(values[3],'->',y_dict[values[0]])
            cnt+=1
        f.write('%s,%s,%s,%s\n'%(values[0],values[1],values[2],y_dict[values[0]]))
print('输出完毕,更新条数：',cnt)