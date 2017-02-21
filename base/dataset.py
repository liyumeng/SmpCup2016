from gensim.models.word2vec import Word2Vec
import pickle
import os
from os.path import dirname

#---------配置项------------------

#项目根目录文件夹名称
project_name='DUTIRTone'

__cur=os.path.abspath('.')
project_path=__cur[:__cur.rindex(project_name)+len(project_name)]

#data所在路径
smp_path=project_path+'/data'

#特征文件输出路径
feature_path=smp_path+'/feature_data'

#结果文件的存储路径
submission_path=project_path+'/submission'

#-------配置项结束-----------------

def load_v1():
    return pickle.load(open(feature_path+'/features.v1.pkl','rb'))

def load_v2():
    return pickle.load(open(feature_path+'/features.v2.pkl','rb'))

'''粉丝数'''
def load_links_dict():
    link_dict={}
    with open(smp_path+'/raw_data/train/train_links.txt') as f:
        items=[item.strip().split(' ') for item in f]
        for item in items:
            link_dict[item[0]]=len(set(item[1:]))
            
    with open(smp_path+'/raw_data/valid/valid_links.txt') as f:
        items=[item.strip().split(' ') for item in f]
        for item in items:
            link_dict[item[0]]=len(set(item[1:]))
    
    return link_dict

def load_w2v(dim=300):
    if dim==200:
        return Word2Vec.load(smp_path+'/word2vec/smp.w2v.200d')
    if dim==300:
        return Word2Vec.load(smp_path+'/word2vec/smp.w2v.300d')
    return None


def load_glove(dim=50):
    return Word2Vec.load(smp_path+'/glove/smp.glove.%dd'%dim)
