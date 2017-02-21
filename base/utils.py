import math
import random
import scipy
from scipy.sparse.csr import csr_matrix
from scipy import sparse
import numpy as np
from base.dataset import load_w2v,smp_path

'''
工具函数
'''
def get(items,i):
    return [item[i] for item in items]

'''载入资源'''
def load_keywords():
    with open(smp_path+'/user_data/keywords.txt',encoding='utf8') as f:
        items=[item.strip() for item in f.readlines()]
    return items

'''计算两个词集余弦的相似度'''
def cal_similar(words1,words2):
    dict1={}
    dict2={}
    for w in words1:
        dict1[w]=dict1.get(w,0)+1
    
    total1=0
    for w in dict1:
        total1+=dict1[w]*dict1[w]
        
    for w in words2:
        dict2[w]=dict2.get(w,0)+1
    total2=0
    for w in dict2:
        total2+=dict2[w]*dict2[w]
        
    res=0
    for w in dict1:
        if w in dict2:
            res+=dict1[w]*dict2[w]/math.sqrt(total1*total2)
    return res


'''去除高度相似的微博'''
def remove_duplicate(sentences):
    random.seed(86)
    max_cnt=100
    res=[]
    for sen in sentences:
        items=sen.split()
        data=[]
        for i in range(len(items)):
            is_similar=False
            for j in range(len(data)):
                if cal_similar(items[i],data[j])>0.9:
                    is_similar=True
                    break
            if is_similar==False:
                data.append(items[i])
        if len(data)>max_cnt:
            random.shuffle(data)
            data=data[:max_cnt]
        res.append(data)

    return [' '.join(item) for item in res]



def merge(fs,filter_indexs=None):
    '''
    传入多个特征的集合，每个特征由[train,test]这样的list组成
    filter_indexs=(filter_index,train_index,valid_index)
    其中filter_index用来预先排除训练集中有缺陷的数据
    train_index和valid_index分别为训练集和验证集的索引值
    '''
    print(type(fs[0][0]))
    tmp=[]
    for f in fs:
        if f[0].ndim==1:
            f=[f[0].reshape(f[0].shape[0],1),f[1].reshape(f[1].shape[0],1)]
        if filter_indexs!=None:
            if len(f)==2:
                f=[f[0][filter_indexs[0]],f[1]]
                f=[f[0][filter_indexs[1]],f[0][filter_indexs[2]],f[1]]
        tmp.append(f)
    
    '''判断是train_data,test_data还是train_data,valid_data,test_data'''
    colCnt=len(tmp[0])
    if type(tmp[0][0])==scipy.sparse.csr.csr_matrix:
        res=[sparse.hstack(tuple([item[i] for item in tmp])) for i in range(colCnt)]
    else:
        res=[np.hstack(tuple([item[i] for item in tmp])) for i in range(colCnt)]
    print(res[0].shape)
    return res


'''
显示各个类别的准确率
'''
def describe(py,ry,detail=False):
    if np.array(py).ndim>1:
        py=[np.argmax(y) for y in py]
    
    right_cnt=len([y for r,y in zip(ry,py) if y==r])
    total_cnt=len(ry)
    res=right_cnt/total_cnt
    print('准确率：',res,'(%d/%d)'%(right_cnt,total_cnt))
    if(detail):
        keys=list(set(ry))
        for k in keys:
            right_cnt=len([r for r,y in zip(ry,py) if y==r and y==k])
            total_cnt=len([y for y in ry if y==k])
            predict_cnt=len([y for y in py if y==k])
            p=right_cnt/(predict_cnt+0.0000001)
            r=right_cnt/(total_cnt+0.0000001)
            print('类别',k,'准确率=%.4f'%p,'(%d/%d)'%(right_cnt,predict_cnt),
                  '\t召回率=%.4f'%r,'(%d/%d)'%(right_cnt,total_cnt),'\tf=%.4f'%(2*p*r/(p+r+0.00001)))
    return res

'''
传入feature集合整合成特征集
'''
def get_xs(fs):
    if len(fs)==1:
        return fs[0]
            
    if type(fs[0])==csr_matrix:
        return sparse.hstack((fs)).tocsc()
    else:
        tmp=[]
        for f in fs:
            if type(f)==csr_matrix:
                tmp.append(f.toarray())
            else:
                tmp.append(f)
        return np.hstack(tmp)
    
'''
传入单词表，返回词向量集合
'''
def get_word_vectors(words,w2v_model=None):
    if w2v_model==None:
        w2v_model=load_w2v()
    vectors=[]
    cnt=0
    for w in words:
        if w in w2v_model:
            vectors.append(w2v_model[w])
        else:
            vectors.append(np.zeros(w2v_model.vector_size))
            cnt+=1
    print('不在词表中的词数量：',cnt)
    return np.array(vectors)


            
        
        
        
        
        
