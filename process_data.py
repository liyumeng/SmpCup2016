'''
处理原始文本，输出为特征文件
features.v1.pkl
features.v2.pkl
author:yuml
5052909@qq.com
'''
import os,sys
sys.path.append(os.path.abspath('../'))
#检查所需的python包
import theano
import jieba
import keras
import xgboost
import gensim
import numpy as np
#-------------------
import random
import pickle
import os
import datetime
import re
from base.utils import remove_duplicate
from base.dataset import feature_path,smp_path

#原始数据
train_labels=smp_path+r'/raw_data/train/train_labels.txt'
train_status=smp_path+'/raw_data/train/train_status.txt'
test_nolabels=smp_path+r'/raw_data/valid/valid_nolabel.txt'
test_status=smp_path+'/raw_data/valid/valid_status.txt'
#城市映射
location=smp_path+r'/user_data/location.txt'
city_loca=smp_path+r'/user_data/city_loca.dict'
short_prov=smp_path+r'/user_data/short_prov.dict'
city_prov=smp_path+r'/user_data/city_prov.dict'
prov_lat=smp_path+r'/user_data/latitude.dict'

reg_topic=re.compile('# (.*?) #')
reg_at=re.compile('@[\S]+')

train_ids=[] #训练集的id顺序
test_ids=[] #测试集的id顺序

'''
载入地区映射表和城市映射表
'''
def load_location():
    dict={}
    with open(location,encoding='utf8') as f:
        for line in f:
            tmp=line.strip().split(':')
            for item in tmp[1].split(','):
                dict[item]=tmp[0]
    return dict
def load_dict(filename):
    city_dict={}
    with open(filename,encoding='utf8') as f:
        for line in f:
            items=line.strip().split(',')
            city_dict[items[0]]=items[1]
    return city_dict

loca_dict=load_location()
loca_map_dict=load_dict(city_loca)
loca2_map_dict=load_dict(short_prov)  #省份缩写
prov_map_dict=load_dict(city_prov)
lat_map_dict={}

reg_city='|'.join(loca_map_dict.keys())
reg_short_prov='|'.join(loca2_map_dict.keys())

'''载入经纬度'''
with open(prov_lat,encoding='utf8') as f:
    for line in f:
        items=line.strip().split(',')
        lat_map_dict[items[0]]=np.array([items[1],items[2]])

'''
用户
'''
class UserInfo(object):
    def __init__(self):
        self.source=[]
        self.times=[]
        self.weeks=[]
        self.hours=[]
        self.reviewCnt=[]
        self.forwardCnt=[]
        self.content=[]
        self.topics=[]
        self.citys=[]
        self.loca_maps=[]
        self.prov_maps=[]
        self.lat_maps=[]
        self.loca2_maps=[]  #映射地名简写
        self.aver_word_cnt=0
        self.aver_length=0
        self.at=[]
        self.id=''
        self.gender=''
        self.location=''
        self.age=''
        self.prov=''
        self.timeImage=np.zeros(7*24)
        self.timeDict={} #过滤同一时间的微博
        self.stimes=[] #发微博的标准时间
        pass
    
    def parse(self,line):
        items=line.split('||')
        self.id=items[0]
        if items[1]=='m':
            self.gender=1
        elif items[1]=='f':
            self.gender=0
        else:
            self.gender=-999
        year=int(items[2])
        if year<1980:
            self.age=0
        elif year>1989:
            self.age=2
        else:
            self.age=1
        prov=items[3].strip().split(' ')[0]
        self.prov=prov
        if prov=='None':
            self.location='None'
        else:
            self.location=loca_dict.get(prov,'境外')

'''
载入用户的label值
'''
def load_train_dict():
    user_dict={}
    train_ids.clear()
    with open(train_labels,encoding='utf8') as f:
        for line in f:
            u=UserInfo()
            u.parse(line)
            user_dict[u.id]=u
            train_ids.append(u.id)
    return user_dict



#将同一个人的微博拼接在一起

def scan_status(filename):
    with open(filename,encoding='utf8') as f:
        for line in f:
            items=line.strip().split(',')
            items[5]=','.join(items[5:])
            yield items[:6]

def get_week_hour(time_string):
    reg_time1=re.compile('^\d{4}-\d{2}-\d{2} \d{2}:')
    reg_time2=re.compile('^今天 (\d{2}):\d{2}')
    reg_time3=re.compile('^\d{1,2}分钟前$')
    
    ts=reg_time1.match(time_string) 
    if ts!=None:
        t=datetime.datetime.strptime(ts.group(), "%Y-%m-%d %H:")
    else:
        ts=reg_time2.match(time_string) #今天 00:00
        t=datetime.datetime.strptime("2016-06-28 0:", "%Y-%m-%d %H:")
        if ts!=None:
            return 1,int(ts.groups()[0]),t
        else:
            if reg_time3.match(time_string)!=None: #3分钟前 默认是2016年6月28日19点，星期二
                return 1,19,t
            else:
                print(time_string)
                return -1,-1
    return t.weekday(),t.hour,t

#将星期，小时转成向量形式
def cnt_to_vector(cnts,dim):
    v=[0 for _ in range(dim)]
    for c in cnts:
        if c>-1:
            v[c]+=1
    return v

'''
表情特征
'''
import re
reg_emoji=re.compile('\[.{1,8}?\]')

with open(smp_path+'/user_data/emoji.txt',encoding='utf8') as f:
    emoji_set=set([item.strip() for item in f.readlines()])

def get_emoji(sentence):
    emojis=[item.replace(' ','') for item in reg_emoji.findall(sentence)]
    emojis=[item for item in emojis if item in emoji_set]
    return ' '.join(emojis)


#拓展用户特征数量
def extend_users(user_dict,status_file):
    for items in scan_status(status_file):
        if(len(items)>6):
            print(len(items))
        if items[0] not in user_dict:
            continue
        user=user_dict[items[0]]
        user.source.append(items[3])
        user.times.append(items[4])
        user.reviewCnt.append(int(items[1]))
        user.forwardCnt.append(int(items[2]))
        ts=get_week_hour(items[4])
        user.weeks.append(ts[0])
        user.hours.append(ts[1])
        user.stimes.append(ts[2])  #发微博的时间
        user.content.append(items[5])
        user.topics.extend(reg_topic.findall(items[5]))
        user.at.extend(reg_at.findall(items[5]))
        user.emoji=get_emoji(items[5])
        if ts[2] not in user.timeDict:
            user.timeImage[ts[0]*24+ts[1]]+=1
            user.timeDict[ts[2]]=1
        
    for key in user_dict:
        u=user_dict[key]
        u.week_vec=cnt_to_vector(u.weeks,7)
        u.hour_vec=cnt_to_vector(u.hours,24)
        words='\n'.join(u.content).split()
        u.aver_word_cnt=len(words)/len(u.content)
        u.aver_length=len('\n'.join(u.content))/len(u.content)
        
        for w in re.findall(reg_city,'\n'.join(u.content)):
            '''如果出现在地域字典里'''
            p=prov_map_dict[w]
            u.citys.append(w)
            u.loca_maps.append(loca_map_dict[w])
            u.prov_maps.append(p)
            u.lat_maps.append(lat_map_dict[p])
        
        for w in re.findall(reg_short_prov,'\n'.join(u.content)):
            '''如果出现在地名简写字典中'''
            u.loca2_maps.append(loca2_map_dict[w])
            
def get_data_by_dict(user_dict,ids):
    data=[] #train_data or test_data
    for id in ids:
        u=user_dict[id]
        #id, content, gender, age, location, topic, review, forward, source, week, hour,
        #aver_word_cnt, aver_length, citys,city_maps
        #emoji, timeImage, times
        data.append([u.id, '\n'.join(u.content),  u.gender,   u.age,  u.location, u.prov,
                     '\n'.join(u.topics), u.reviewCnt, u.forwardCnt,
                     '\n'.join(u.source),'\n'.join(u.at),u.week_vec,
                      u.hour_vec,      u.aver_word_cnt,   u.aver_length,
                     '\n'.join(u.citys),'\n'.join(u.loca_maps),u.emoji,u.timeImage,u.stimes,
                    '\n'.join(u.prov_maps),np.array(u.lat_maps).astype('float'),'\n'.join(u.loca2_maps)])
    return data

def get_test_data(test_nolabels,test_status):
    test_user_dict={}
    test_ids.clear()
    with open(test_nolabels,encoding='utf8') as f:
        for line in f:
            u=UserInfo()
            u.id=line.strip()
            test_user_dict[u.id]=u
            test_ids.append(u.id)
    extend_users(test_user_dict,test_status)
    test_data=get_data_by_dict(test_user_dict,test_ids)
    return test_data

#----------------------保存数据到features.v1.pkl----------------------------------------------------------

'''
持久化数据
训练集：3200个
测试集：980个
输出到/data/feature_data/features.v1.pkl中
'''

user_dict=load_train_dict()
extend_users(user_dict,train_status)
print('user:%d'%len(user_dict))
train_data=get_data_by_dict(user_dict,train_ids)
test_data=get_test_data(test_nolabels,test_status)
print('train_data:%d, test_data:%d'%(len(train_data),len(test_data)))

abs_path=os.path.abspath(feature_path)
if os.path.exists(abs_path)==False:
    os.mkdir(abs_path)

pickle.dump([train_data,test_data],open(feature_path+'/features.v1.pkl','wb'))
print('train_data:%d, test_data:%d'%(len(train_data),len(test_data)))
print('数据已保存到：%s/features.v1.pkl'%feature_path)
print('...')
#-------------------------继续生成features.v2.pkl-----------------------------------------------------------
import pickle
import numpy as np
from base.dataset import smp_path,feature_path,load_v1,load_links_dict
from base.utils import cal_similar,get,load_keywords,remove_duplicate,get_word_vectors
train_data,test_data=load_v1()

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
'''获得计数类文本特征'''
def get_f_cnter(xs,min_df=5):
    cnter=CountVectorizer(min_df=min_df)
    cnter.fit(xs[0]+xs[1])
    f_xs=[cnter.transform(x) for x in xs]
    return f_xs

'''将整数转换占比例形式'''
def to_rate(a):
    b=np.sum(a,axis=1)
    return a/np.outer(b,np.ones(a.shape[1]))

'''将日期转成int型特征'''
def date_to_int(d):
    return (d.year-2009)*12+d.month-10

#-------------------1. 文本特征-------------------------------------
'''
tfidf类特征
'''

from sklearn.decomposition import TruncatedSVD
from scipy import sparse

content=get(train_data,1),get(test_data,1)
xs=content

'''词tfidf特征'''
tfidf=TfidfVectorizer(min_df=5,ngram_range=(1,2))
tfidf.fit(xs[0])
f_word=[tfidf.transform(x) for x in xs]
print('f_word',f_word[0].shape)

'''字tfidf特征'''
tfidf2=TfidfVectorizer(min_df=5,ngram_range=(1,2),analyzer='char')
tfidf2.fit(xs[0])
f_letter=[tfidf2.transform(x) for x in xs]
print('f_letter',f_letter[0].shape)

'''话题词特征'''
theme=get(train_data,6),get(test_data,6)
xs=theme
tfidf3=TfidfVectorizer(min_df=5,ngram_range=(1,2),analyzer='char')
tfidf3.fit(xs[0]+xs[1])
f_theme_word=[tfidf3.transform(x) for x in xs]

'''话题特征'''
nospace=[[item.replace(' ','') for item in t] for t in theme]
f_theme=get_f_cnter(nospace)

'''话题字特征'''
xs=nospace
tfidf3=TfidfVectorizer(min_df=5,ngram_range=(1,2),analyzer='char')
tfidf3.fit(xs[0]+xs[1])
f_theme_letter=[tfidf3.transform(x) for x in xs]

'''表情特征'''
emoji=get(train_data,17),get(test_data,17)
f_emoji=get_f_cnter(emoji)

'''分享源'''
shared=get(train_data,10),get(test_data,10)
f_shared=get_f_cnter(shared)

'''信息来源'''
source=get(train_data,9),get(test_data,9)
f_source=get_f_cnter(source)

'''关键词表'''
xs=content
keywords=load_keywords()
cnter=CountVectorizer(binary=True,vocabulary=keywords)
cnter.fit(xs[0]+xs[1])
f_keywords=[cnter.transform(x) for x in xs]


'''svd降维'''
svd=TruncatedSVD(n_components=500)
svd.fit(f_word[0])
f_word_pca=svd.transform(f_word[0]),svd.transform(f_word[1])

svd=TruncatedSVD(n_components=500)
svd.fit(f_letter[0])
f_letter_pca=svd.transform(f_letter[0]),svd.transform(f_letter[1])

'''词tfidf特征 ngram1'''
tfidf4=TfidfVectorizer(min_df=3,ngram_range=(1,1))
tfidf4.fit(xs[0])
f_word_n1=[tfidf4.transform(x) for x in xs]
print('f_word_n1',f_word_n1[0].shape)

'''字tfidf特征 ngram1'''
tfidf5=TfidfVectorizer(min_df=3,ngram_range=(1,1),analyzer='char')
tfidf5.fit(xs[0])
f_letter_n1=[tfidf5.transform(x) for x in xs]
print('f_letter_n1',f_letter_n1[0].shape)

'''source tfidf 特征'''
source=get(train_data,9),get(test_data,9)
tfidf6=TfidfVectorizer(min_df=3,ngram_range=(1,1))
tfidf6.fit(source[0])
f_source_tfidf=[tfidf6.transform(x) for x in source]

f_text=[f_word,f_letter,f_word_pca,f_letter_pca,f_theme,f_theme_word,f_theme_letter,
       f_emoji,f_shared,f_source,f_keywords,f_word_n1,f_letter_n1,f_source_tfidf]

#-------------------2. 统计特征-------------------------------------------------------
import numpy as np
'''统计类特征'''

'''粉丝数'''
links_dict=load_links_dict()

def aver(items):
    if len(items)==0:
        return 0
    return np.average(items)

def get_f_statistic(user_data):
    fs=[]
    for item in user_data:
        sens=item[1].split('\n')
        '''去除与自己重复的'''
        norSens=remove_duplicate(sens)
        
        '''去除分享后'''
        noShared=[]
        for source,sen in zip(item[9],sens):
            if source.find('分享') or sen.find('分享'):
                continue
            noShared.append(sen)

        cnt=len(sens)
        norCnt=len(norSens)
        nosCnt=len(noShared)
        rrate=(cnt-norCnt)/cnt
        
        review=np.array(item[7])
        forward=np.array(item[8])
        
        reviewCnt=np.sum(review)
        forwardCnt=np.sum(forward)
        
        averReview=aver(review)
        averForward=aver(forward)
        
        goodCnt=np.sum((review>4)+(forward>4))
        goodRate=1.0*goodCnt/cnt
        
        averLetter=aver([len(sen) for sen in sens])
        averWord=aver([len(sen.split()) for sen in sens])

        norAverLetter=aver([len(sen) for sen in norSens])
        norAverWord=aver([len(sen.split()) for sen in norSens])
        
        nosAverLetter=aver([len(sen) for sen in noShared])
        nosAverWord=aver([len(sen.split()) for sen in noShared])

        linksCnt=links_dict.get(item[0],0)
        
        fs.append([cnt,norCnt,nosCnt,rrate,reviewCnt,forwardCnt,
                   averReview,averForward,goodCnt,goodRate,
                   averLetter,averWord,norAverLetter,norAverWord,
                  nosAverLetter,nosAverWord,linksCnt])
        
    return np.array(fs)


f_stat=get_f_statistic(train_data),get_f_statistic(test_data)

#---------------------3. 时间特征------------------------------------------------
from collections import Counter
import datetime
'''时间类特征'''
def get_f_times_by_user(user_item):
    times=sorted(user_item[19])
    '''连续活跃小时数特征'''
    hourCnt=1
    longHourCnt=0
    for i in range(1,len(times)):
        if (times[i]-times[i-1]).seconds<3600*3:
            hourCnt+=1
            if hourCnt>longHourCnt:
                longHourCnt=hourCnt
        else:
            hourCnt=1
            
    '''连续活跃天数特征'''
    dayCnt=1
    longDayCnt=0
    for i in range(1,len(times)):
        if (times[i]-times[i-1]).days<2:
            if times[i].day-times[i-1].day==1:
                dayCnt+=1
                if dayCnt>longDayCnt:
                    longDayCnt=dayCnt
        else:
            dayCnt=1

    hourDict=Counter([t.strftime('%Y%m%d%H') for t in times])
    dayDict=Counter([t.strftime('%Y%m%d') for t in times])
    monthDict=Counter([t.strftime('%Y%m') for t in times])

    norHourCnt=len(hourDict)
    norDayCnt=len(dayDict)
    norMonthCnt=len(monthDict)
    
    maxHours=np.max(list(hourDict.values()))
    maxDays=np.max(list(dayDict.values()))
    
    ftCnt=np.array(user_item[18])
    ftNormCnt=ftCnt/np.sum(ftCnt)
    fs=np.array([longHourCnt,longDayCnt,norHourCnt,
                 norDayCnt,norMonthCnt,maxHours,maxDays])
    return np.concatenate((ftCnt,ftNormCnt,fs))

def get_f_times(user_data):
    fs=[]
    for user_item in user_data:
        f=get_f_times_by_user(user_item)
        f_week=np.array(user_item[11])
        f_hour=np.array(user_item[12])

        f_norm_week=f_week/np.sum(f_week)
        f_norm_hour=f_hour/np.sum(f_hour)
        
        
        fs.append(np.concatenate((f_week,f_norm_week,f_hour,f_norm_hour,f)))
    return np.array(fs)

f_times=get_f_times(train_data),get_f_times(test_data)

'''统计每月发微博数量的分布'''
def get_time_cnt(times):
    data=[]
    for t in times:
        data.append([(d.year-2009)*12+d.month-10 for d in t])
    data=[' '.join(map(str,item)) for item in data]
    vocab=[str(i) for i in range(81)]
    cnter=CountVectorizer(min_df=0,token_pattern='(?u)\\b\\w+\\b',vocabulary=vocab)
    return cnter.fit_transform(data).toarray()
    

times=get(train_data,19),get(test_data,19)
f_monthDist=get_time_cnt(times[0]+times[1])
f_monthDist=f_monthDist[:len(times[0]),:],f_monthDist[len(times[0]):,:]

'''统计最后一次发微博距今多少天'''
tday=datetime.datetime.strptime("2016-06-28 0:", "%Y-%m-%d %H:")

def get_last_day(times):
    f=[tday-np.max(t) for t in times]
    f=[t.days for t in f]
    f=np.array(f).reshape((len(f),1))
    return f

f_last_day=get_last_day(times[0]),get_last_day(times[1])

f_times=np.hstack((f_times[0],f_monthDist[0],f_last_day[0])),np.hstack((f_times[1],f_monthDist[1],f_last_day[1]))


print('时间类特征：',f_times[0].shape)
print('...')
#--------------------4. 地名特征 ----------------------------
'''城市映射'''
citys=get(train_data,15),get(test_data,15)
loca_maps=get(train_data,16),get(test_data,16)
prov_maps=get(train_data,20),get(test_data,20)
lat_maps=get(train_data,21),get(test_data,21)
short_prov_maps=get(train_data,22),get(test_data,22)

fp_citys=get_f_cnter(citys,1)
fp_loca_maps=get_f_cnter(loca_maps,1)
fp_prov_maps=get_f_cnter(prov_maps,1)
fp_exist=[(np.array([len(c) for c in items])>0).astype('int') for items in citys]
fp_loca2_maps=get_f_cnter(short_prov_maps,1)


fp_exist=[item.reshape(item.shape[0],1) for item in fp_exist]


def get_fp_lat(user_data):
    fs=[]
    for item in user_data:
        if len(item)==0:
            fs.append([0,0])
        else:
            fs.append(np.average(item,axis=0))
    return np.array(fs)
fp_lat_maps=get_fp_lat(lat_maps[0]),get_fp_lat(lat_maps[1])

fp=[fp_citys,fp_loca_maps,fp_prov_maps,fp_exist,fp_lat_maps,fp_loca2_maps]

#----------------------5. 抽取y值--------------------------------------
'''抽取y值'''
ids=get(train_data,0),get(test_data,0)

y_gen=get(train_data,2)
y_age=get(train_data,3)
y_loca=get(train_data,4)

loca_enum='None,华北,华东,华南,西南,华中,东北,西北,境外'.split(',')
y_loca=[loca_enum.index(y)-1 for y in y_loca]
            
ys=np.array([y_gen,y_age,y_loca])

'''train 与 test特征分开输出'''
f_text_train=[item[0] for item in f_text]
f_text_test=[item[1] for item in f_text]

fp_train=[item[0] for item in fp]
fp_test=[item[1] for item in fp]

f_train=[f_text_train,f_stat[0],f_times[0],fp_train]
f_test=[f_text_test,f_stat[1],f_times[1],fp_test]

f_content=(get(train_data,1),get(test_data,1))
pickle.dump([ids,ys,f_train,f_test,f_content],open(feature_path+'/features.v2.pkl','wb'))
print('features.v2.pkl输出完毕!')
print('...')
#-----------------------6. 输出word2vec---------------------------------------

'''word2vec'''
from sklearn.feature_extraction.text import TfidfVectorizer
from base.dataset import load_w2v
import random
def filter_content(content):
    return content.replace('\xa0',' ')
    
def split_weibo(contents):
    sens=[]
    ids=[]
    for item in contents:
        c_sens=filter_content(item).split('\n')
        ids.append(range(len(sens),len(sens)+len(c_sens)))
        sens.extend(c_sens)
    return sens,ids
'''
将所有句子重新分配给每个人
'''
def get_f_w2v(xs,ids,vector_size=300):
    x_cnn=[]
    max_dim=100
    for indexs in ids:
        item=[]
        for i in indexs[:min(max_dim,len(indexs))]:
            item.append(xs[i])
        if len(item)<max_dim:
            for i in range(max_dim-len(item)):
                item.append(np.zeros(vector_size))
        #random.shuffle(item)
        x_cnn.append([np.array(item)])
    return np.array(x_cnn)

IsRefresh=True
import os
import pickle
filename=feature_path+'/f_w2v_tfidf.300.cache'
if os.path.exists(filename)==False or IsRefresh:
    '''split to sentences'''
    sens,tids=split_weibo(f_content[0])

    tfidf=TfidfVectorizer(min_df=3,)
    f_tfidf_raw=tfidf.fit_transform(sens)
    print('tfidf dim:',f_tfidf_raw.shape)

    '''get word2vec library'''
    model=load_w2v(300)
    vectors=get_word_vectors(tfidf.get_feature_names(),model)
    f_flatten_w2v=f_tfidf_raw*vectors

    '''get feature'''
    f_w2v=get_f_w2v(f_flatten_w2v,tids,model.vector_size)
    print('f_w2v dim:',f_w2v.shape)

    test_sens,test_ids=split_weibo(f_content[1])

    f_tfidf_raw_test=tfidf.transform(test_sens)

    f_flatten_w2v_test=f_tfidf_raw_test*vectors

    f_w2v_test=get_f_w2v(f_flatten_w2v_test,test_ids,model.vector_size)

    pickle.dump([ids,f_w2v,f_w2v_test],open(filename,'wb'))

else:    
    fids,f_w2v,f_w2v_test=pickle.load(open(filename,'rb'))
    
print(filename,'输出完毕')
print('...')
#--------------------7. 输出降维的tfidf特征---------------------------
'''word svd300'''
def get_f_cnn(xs,ids,vector_size=300):
    x_cnn=[]
    max_dim=100
    for indexs in ids:
        item=[]
        for i in indexs[:min(max_dim,len(indexs))]:
            item.append(xs[i])
        if len(item)<max_dim:
            for i in range(max_dim-len(item)):
                item.append(np.zeros(vector_size))
        #random.shuffle(item)
        x_cnn.append([np.array(item)])
    return np.array(x_cnn)

filename=feature_path+'/f_word_svd.300.cache'

sens,tids=split_weibo(f_content[0]+f_content[1])

tfidf=TfidfVectorizer(min_df=3,)
f_tfidf=tfidf.fit_transform(sens)
print('tfidf dim:',f_tfidf.shape)

svd=TruncatedSVD(n_components=300)
f_svd=svd.fit_transform(f_tfidf)
f_cnn=get_f_cnn(f_svd,tids)
pickle.dump([ids,f_cnn[:3200],f_cnn[3200:]],open(filename,'wb'))
    
print(filename,'输出完毕')
print('...')
#--------------------------8. 输出字的tfidf降维特征------------------------
'''letter svd300'''
filename=feature_path+'/f_letter_svd.300.cache'

sens,tids=split_weibo(f_content[0]+f_content[1])

tfidf=TfidfVectorizer(min_df=3,analyzer='char')
f_tfidf=tfidf.fit_transform(sens)
print('tfidf dim:',f_tfidf.shape)

svd=TruncatedSVD(n_components=300)
f_svd=svd.fit_transform(f_tfidf)
f_cnn=get_f_cnn(f_svd,tids)
pickle.dump([ids,f_cnn[:3200],f_cnn[3200:]],open(filename,'wb'))
    
print(filename,'输出完毕')
print('...')

#--------------------------9. 输出来源的tfidf降维特征---------------------------
'''source svd300'''
f_source=get(train_data,9)+get(test_data,9)
filename=feature_path+'/f_source_svd.300.cache'

sens,tids=split_weibo(f_source)

tfidf=TfidfVectorizer(min_df=3,analyzer='char')
f_tfidf=tfidf.fit_transform(sens)
print('tfidf dim:',f_tfidf.shape)

svd=TruncatedSVD(n_components=300)
f_svd=svd.fit_transform(f_tfidf)
f_cnn=get_f_cnn(f_svd,tids)
pickle.dump([ids,f_cnn[:3200],f_cnn[3200:]],open(filename,'wb'))
    
print(filename,'输出完毕')
print('...')
#------------------------------10. 抽取来源的地名特征--------------------------
sources=get(train_data+test_data,9)
from base.dataset import smp_path
city_loca=smp_path+r'/user_data/city_loca.dict'
def load_dict(filename):
    city_dict={}
    with open(filename,encoding='utf8') as f:
        for line in f:
            items=line.strip().split(',')
            city_dict[items[0]]=items[1]
    return city_dict
loca_map_dict=load_dict(city_loca)

city_prov=smp_path+r'/user_data/city_prov.dict'
prov_map_dict=load_dict(city_prov)
import re
fp_s_loca=[]
fp_s_prov=[]
fp_s_exist=[]

for s in sources:
    citys=[]
    provs=[]
    locas=[]
    for key in loca_map_dict:
        citys.extend(re.findall(key,s))
    for city in citys:
        provs.append(prov_map_dict[city])
        locas.append(loca_map_dict[city])
    if len(citys)>0:
        fp_s_exist.append(1)
    else:
        fp_s_exist.append(0)
    fp_s_loca.append(' '.join(locas))
    fp_s_prov.append(' '.join(provs))
    
def get_f_cnter2(xs,min_df=1):
    cnter=CountVectorizer(min_df=min_df)
    return cnter.fit_transform(xs)

fp_s_loca=get_f_cnter2(fp_s_loca,1)
fp_s_prov=get_f_cnter2(fp_s_prov,1)
fp_s_exist=np.array(fp_s_exist).reshape((len(fp_s_exist),1))
ids=get(train_data+test_data,0)
pickle.dump([ids,fp_s_loca,fp_s_prov,fp_s_exist],open(feature_path+'/loca.source.feature','wb'))
#----------------------------------以句子为单位进行特征输出---------------------------------------------------

# data中存储了每个用户的数据
# indexes中存储了用户对应的位置信息
data=[]
fids=[]
indexes=[]
for item in train_data+test_data:
    id=item[0]
    fids.append(id)
    sens=item[1].split('\n')
    reviews=item[7]
    forwards=item[8]
    sources=item[9].split('\n')
    times=item[19]
    years=[t.year for t in times]
    days=[t.day for t in times]
    months=[t.month for t in times]
    weeks=[t.weekday() for t in times]
    hours=[t.hour for t in times]
    begin=len(data)
    data.extend(list(zip(sens,reviews,forwards,sources,years,months,days,weeks,hours)))
    indexes.append(range(begin,len(data)))
    
#关键词特征
f_keys=[]
import re
rules=['#','分享','http','地图']
for item in data:
    val=np.zeros((len(rules),))
    for i,rule in enumerate(rules):
        if re.search(rule,item[0])!=None:
            val[i]=1
    f_keys.append(val)
f_keys=np.array(f_keys)

#count feature
f_cnt=[]
for item in data:
    f_cnt.append([item[1],item[2]])
f_cnt=np.array(f_cnt)

#time feature
from keras.utils.np_utils import to_categorical
f_week=to_categorical(get(data,7))
f_hour=to_categorical(get(data,8))

#char feature
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(min_df=3)
f_char=tfidf.fit_transform(get(data,0))
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=300-37)
f_char_svd=svd.fit_transform(f_char)

fs=np.hstack([f_keys,f_cnt,f_week,f_hour,f_char_svd])
f_cnn=[]
for index in indexes:
    f=fs[index]
    if f.shape[0]<100:
        t=np.vstack((f,np.zeros((100-f.shape[0],300))))
    else:
        t=f[:100]
    f_cnn.append(t)
f_cnn=np.array(f_cnn)
pickle.dump([fids,f_cnn],open(feature_path+'/f_sens.300.pkl','wb'))
#-------------------------------- 程序运行完毕---------------------------------------------------
print('程序运行完毕')
