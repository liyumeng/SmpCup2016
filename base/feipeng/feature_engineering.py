'''
特征工程：数据预处理
'''
import numpy as np
import pandas as pd
import os
import re
import jieba
import pickle
import codecs

class features(object):
	def __init__(self, inpaths, outpaths):
		print('---加载数据---')
		# 训练集
		self.train_info = inpaths['train_info']
		self.train_labels = inpaths['train_labels']
		self.train_links = inpaths['train_links']
		self.train_status = inpaths['train_status']
		# 测试集
		self.test_info = inpaths['test_info']
		self.test_nolabels = inpaths['test_nolabels']
		self.test_links = inpaths['test_links']
		self.test_status = inpaths['test_status']
		# 停用词
		self.stopwords = inpaths['stopwords']
		
		# 中间文件
		self.status_file = outpaths['status_file']
		self.text_file = outpaths['text_file']
		# 最终文件
		self.data_file = outpaths['data_file']
		self.features_file = outpaths['features_file']
		print('---加载数据---')
		
		self.new_labels = inpaths['new_labels']
		labels = []
		with codecs.open(self.train_labels, 'r', encoding='utf-8')as f:
			for line in f:
				line = line.replace('||',',')
				labels.append(line)
		with codecs.open(self.new_labels, 'w', encoding='utf-8')as f:
			for line in labels:
				f.write(line)
		self.train_labels = self.new_labels

		# 开始构造训练集和测试集的特征
		self.df = pd.read_csv(self.train_labels, names=['uid','gender','birthday','location'], encoding='utf-8')
		train_x = pd.DataFrame(self.df.uid)
		test_x = pd.read_csv(self.test_nolabels, encoding='utf-8', names=['uid'])
		# 合并
		self.data_x = pd.concat([train_x,test_x], axis=0, ignore_index=True)
		print ('训练集+测试集共有{}个样本'.format(self.data_x.shape[0]))
		self.features = self.data_x[:]  # 特征

	def build(self):
		print('---处理微博status文件---')
		self.process_status()
		print('---建立统计特征---')
		self.build_feature()
		print('---处理文本特征---')
		self.process_text()
		print('---处理粉丝信息---')
		self.process_fans()
		print('---输出完毕---')
		pickle.dump(self.data_x, open(self.data_file,'wb'))
		pickle.dump(self.features, open(self.features_file, 'wb'))


	def process_status(self):
		if os.path.isfile(self.status_file):
		    os.remove(self.status_file)
		if os.path.isfile(self.text_file):
		    os.remove(self.text_file)

		paths = [self.train_status, self.test_status]
		for path in paths:
		    with codecs.open(path,'r',encoding='utf-8') as f,\
		        codecs.open(self.status_file,'a',encoding='utf-8') as status_out,\
		        codecs.open(self.text_file,'a',encoding='utf-8') as text_out:
		            i = 0
		            for line in f:
		                item = line.strip().split(',',5)

		                status_info = item[:5]
		                if len(status_info) != 5:
		                    continue
		                status_out.write(','.join(status_info)+'\n')
		                text_info = item[5]
		                text_out.write(text_info+'\n')
		                i+=1
		            print ('{}条微博'.format(i))
		self.status = pd.read_csv(self.status_file, names=['uid','retweet','review','source','time'])
		self.text = pd.read_csv(self.text_file, names=['content'], sep='delimiter')

		if not self.text.shape[0] == self.status.shape[0]:
		    print ('status 和 text 不匹配!!!')

		self.text['uid'] = self.status.uid
		self.text = self.text[['uid','content']]

		# 除去多余的微博
		self.status = self.status[self.status.uid.apply(lambda x:x in self.data_x.uid.values)]
		self.text = self.text[self.text.uid.apply(lambda x:x in self.data_x.uid.values)]

		'''
		处理时间
		微博时间分为三种格式: 
		1、2015-11-10 09:13:35
		2、今天 00:15
		3、7分钟前
		经过统计，将2中‘今天’替换为‘2016-06-28’，方便计算
		最后一条微博的时间为“2016-06-28 22:32:00”， 将X分钟前设为“2016-06-28 23:00:00”
		有6条错误格式的时间“2014-06-12 00:25:45 来自”。  去掉“来自”
		'''
		def processtime(x):
		    reg_time1 = re.compile('^\d{4}-\d{2}-\d{2} \d{2}:')
		    reg_time2 = re.compile('^\d{1,2}分钟前$')
		    pattern1 = reg_time1.match(x)
		    if not pattern1:
		        if reg_time2.match(x):
		            x = '2016-06-28 23:00:00'
		        x = x.replace('今天', '2016-06-28')
		    return x[:19]

		# 处理时间
		self.status['time'] = self.status['time'].map(lambda x: processtime(x))
		self.status['time'] = self.status['time'].map(lambda x: pd.to_datetime(x, errors='coerce'))
		self.status['date'] = self.status.time.map(lambda x: x.strftime('%Y-%m-%d'))
		self.status['week'] = self.status['time'].map(lambda x: x.dayofweek)
		self.status['hour'] = self.status['time'].map(lambda x: x.hour)
		# 统计微博词数
		self.text['word_count'] = self.text.content.apply(lambda x: len(str(x).strip().split()))
		self.text['source'] = self.status.source


	def build_feature(self):
		# 微博总数
		temp = pd.DataFrame(self.status.groupby('uid').size(),columns=['weibo_count']).reset_index()
		self.features = self.features.merge(temp[['uid','weibo_count']], how='left',on='uid')

		# 微博去重总数
		temp = pd.DataFrame(self.text.drop_duplicates().groupby('uid').size(),columns=['weibo_unique_count']).reset_index()
		self.features = self.features.merge(temp[['uid','weibo_unique_count']], how='left',on='uid')

		# 转发评论数
		temp = self.status.groupby('uid').sum().reset_index().rename(columns={'retweet':'retweet_count','review':'review_count'})
		self.features = self.features.merge(temp[['uid', 'retweet_count', 'review_count']], how='left', on='uid')

		# 带转发的微博数、带评论的微博数
		temp = pd.DataFrame(self.status[self.status.retweet > 0].groupby('uid').size(),columns=['retweet_weibo_count']).reset_index()
		self.features = self.features.merge(temp[['uid','retweet_weibo_count']], how='left',on='uid')
		temp = pd.DataFrame(self.status[self.status.review > 0].groupby('uid').size(),columns=['review_weibo_count']).reset_index()
		self.features = self.features.merge(temp[['uid','review_weibo_count']], how='left',on='uid')
		self.features.fillna(0, inplace=True)

		# 平均转发、评论数
		self.features['retweet_average_count'] = self.features.retweet_count / self.features.weibo_count
		self.features['review_average_count'] = self.features.review_count / self.features.weibo_count

		# 微博转发率（有转发的微博/微博总数） retweet_rate
		# 微博评论率（有评论的微博/微博总数） review_rate
		self.features['retweet_rate'] = self.features.retweet_weibo_count / self.features.weibo_count
		self.features['review_rate'] = self.features.review_weibo_count / self.features.weibo_count


		# 来源总数
		temp = pd.DataFrame(self.status.groupby('uid').source.nunique()).reset_index().rename(columns={'source':'source_count'})
		self.features = self.features.merge(temp[['uid','source_count']], how='left',on='uid')

		# 微博登录天数 day_post_count
		temp = pd.DataFrame(self.status.groupby('uid').date.nunique()).reset_index().rename(columns={'date':'day_post_count'})
		self.features = self.features.merge(temp[['uid','day_post_count']], how='left',on='uid')

		# 微博总天数  day_total_count
		temp = pd.DataFrame(((pd.to_datetime((self.status.groupby('uid').date.max())) - \
		             pd.to_datetime((self.status.groupby('uid').date.min())))/ np.timedelta64(1, 'D')).astype(float)\
		            ).reset_index().rename(columns={'date':'day_total_count'})
		self.features = self.features.merge(temp[['uid','day_total_count']], how='left',on='uid')

		# 活跃天数比(发微博的天数/（最后一天-第一天）) day_rate
		self.features['day_rate'] = self.features.day_post_count / self.features.day_total_count

		# 日均微博数  everyday_weibo_count
		self.features['everyday_weibo_count'] = self.features.weibo_count / self.features.day_post_count

		# 总词数  word_total_count
		# 微博平均词数   word_average_count
		temp = pd.DataFrame(self.text.groupby('uid').word_count.sum()).reset_index().rename(columns={'word_count':'word_total_count'})
		self.features = self.features.merge(temp[['uid','word_total_count']], how='left',on='uid')
		self.features['word_average_count'] = self.features.word_total_count / self.features.weibo_count

		# 周几的微博
		for i in range(7):
		    temp = pd.DataFrame(self.status[self.status.week == i].groupby('uid').size(),columns=['weibo_count_of_week{}'.format(i+1)]).reset_index()
		    self.features = self.features.merge(temp[['uid','weibo_count_of_week{}'.format(i+1)]], how='left',on='uid')
		self.features.fillna(0, inplace=True)

		# 工作日和周末的微博
		self.features['weibo_count_of_workday'] = (self.features.weibo_count_of_week1 + self.features.weibo_count_of_week2 + self.features.weibo_count_of_week3 +\
		                                      self.features.weibo_count_of_week4 + self.features.weibo_count_of_week5)
		self.features['weibo_count_of_weekend'] = (self.features.weibo_count_of_week6 + self.features.weibo_count_of_week7)

		# 各天微博的比例
		for i in range(1,8):
		    self.features['weibo_rate_of_week{}'.format(i)] = (self.features['weibo_count_of_week{}'.format(i)] / self.features.weibo_count)
		self.features['weibo_rate_of_workday'] = self.features.weibo_count_of_workday / self.features.weibo_count
		self.features['weibo_rate_of_weekend'] = self.features.weibo_count_of_weekend / self.features.weibo_count

		# 每天各个时段的微博数量
		temp = pd.DataFrame(self.status[self.status.hour.apply(lambda x: x in range(0,6))].groupby(self.status.uid).size(),\
		                    columns=['weibo_count_of_midnight']).reset_index()
		self.features = self.features.merge(temp[['uid','weibo_count_of_midnight']], how='left',on='uid')
		temp = pd.DataFrame(self.status[self.status.hour.apply(lambda x: x in range(6,12))].groupby(self.status.uid).size(),\
		                    columns=['weibo_count_of_morning']).reset_index()
		self.features = self.features.merge(temp[['uid','weibo_count_of_morning']], how='left',on='uid')
		temp = pd.DataFrame(self.status[self.status.hour.apply(lambda x: x in range(12,18))].groupby(self.status.uid).size(),\
		                    columns=['weibo_count_of_afternoon']).reset_index()
		self.features = self.features.merge(temp[['uid','weibo_count_of_afternoon']], how='left',on='uid')
		temp = pd.DataFrame(self.status[self.status.hour.apply(lambda x: x in range(18,24))].groupby(self.status.uid).size(),\
		                    columns=['weibo_count_of_night']).reset_index()
		self.features = self.features.merge(temp[['uid','weibo_count_of_night']], how='left',on='uid')
		self.features.fillna(0, inplace=True)

		# 各个时段的微博比例
		self.features['weibo_rate_of_midnight'] = self.features.weibo_count_of_midnight / self.features.weibo_count
		self.features['weibo_rate_of_morning'] = self.features.weibo_count_of_morning / self.features.weibo_count
		self.features['weibo_rate_of_afternoon'] = self.features.weibo_count_of_afternoon / self.features.weibo_count
		self.features['weibo_rate_of_night'] = self.features.weibo_count_of_night / self.features.weibo_count

		# 各小时的微博数量
		for i in range(24):
		    temp = pd.DataFrame(self.status[self.status.hour == i].groupby('uid').size(),columns=['weibo_count_of_hour{}'.format(i)]).reset_index()
		    self.features = self.features.merge(temp[['uid','weibo_count_of_hour{}'.format(i)]], how='left',on='uid')
		self.features.fillna(0, inplace=True)
		for i in range(24):
		    self.features['weibo_rate_of_hour{}'.format(i)] = (self.features['weibo_count_of_hour{}'.format(i)] / self.features.weibo_count)

		# 按时间分段 间隔3小时
		for i in range(0,24,3):
		    temp = pd.DataFrame(self.status[self.status.hour.apply(lambda x: x in range(i,i+3))].groupby(self.status.uid).size(),\
		                    columns=['weibo_count_of_{}_plus3'.format(i)]).reset_index()
		    self.features = self.features.merge(temp[['uid','weibo_count_of_{}_plus3'.format(i)]], how='left',on='uid')
		self.features.fillna(0, inplace=True)
		for i in range(0,24,3):
		    self.features['weibo_rate_of_{}_plus3'.format(i)] = (self.features['weibo_count_of_{}_plus3'.format(i)] / self.features.weibo_count)
		    
		    
		# 替换掉inf值
		self.features = self.features.replace(np.inf, 0)
		self.features.fillna(0, inplace=True)

		del temp


	def process_text(self):
		# 停用词
		with codecs.open(self.stopwords, encoding='utf-8', errors='ignore')as f:
		    stop =set()
		    for line in f:
		        stop.add(line.strip())        
		def de_stop(line):
		    line = line.strip().split()
		    res = []
		    for word in line:
		        if word not in stop:
		            res.append(word)
		    return ' '.join(res)
		# 去停用词的微博正文
		self.text['words'] = self.text.content.apply(lambda x: de_stop(x))
		temp = pd.DataFrame(self.text.groupby('uid')['words'].apply(lambda x: ' '.join(x))).reset_index()
		self.data_x = self.data_x.merge(temp[['uid','words']], how='left',on='uid')
		# source信息
		temp = pd.DataFrame(self.status.groupby('uid')['source'].apply(lambda x: ' '.join(x))).reset_index()
		self.data_x = self.data_x.merge(temp[['uid','source']], how='left',on='uid')
		# source分词
		def fenci(line):
		    line = line.strip().split()
		    line = ''.join(line)
		    seglist = jieba.cut(line)
		    line = ' '.join(seglist)
		    return line
		self.data_x['source_fenci'] = self.data_x.source.apply(lambda x:fenci(x))
		self.data_x['weibo_and_source'] = (self.data_x.words + self.data_x.source_fenci)


	def process_fans(self):
		with open(self.train_links) as f:
		    res = []
		    for line in f:
		        items = line.strip().split()
		        uid = int(items[0])
		        fans = ' '.join(items[1:])
		        number = len(fans.split())
		        res.append([uid, fans, number])
		trainfans = pd.DataFrame(res, columns=['uid','fans','count_of_fans'])

		with open(self.test_links) as f:
		    res = []
		    for line in f:
		        items = line.strip().split()
		        uid = int(items[0])
		        fans = ' '.join(items[1:])
		        number = len(fans.split())
		        res.append([uid, fans, number])
		testfans = pd.DataFrame(res, columns=['uid','fans','count_of_fans'])

		fans = pd.concat([trainfans, testfans], axis=0)
		fans.drop_duplicates(inplace=1)

		self.data_x = self.data_x.merge(fans[['uid','fans','count_of_fans']], how='left',on='uid')
		self.data_x['has_fans'] = 0
		self.data_x['has_fans'][self.data_x.fans.notnull()] = 1
		self.data_x.fillna(0, inplace=True)

		# fans info
		self.features['has_fans'] = self.data_x.has_fans
		self.features['count_of_fans'] = self.data_x.count_of_fans

		self.features['weibo_per_fans'] = self.features.weibo_count / self.features.count_of_fans
		self.features['ret_per_fans'] = self.features.retweet_count / self.features.count_of_fans
		self.features['rev_per_fans'] = self.features.review_count / self.features.count_of_fans

		self.features = self.features.replace(np.inf, 0)
		self.features.fillna(0, inplace=True)

		self.features['fans_0_50'] = 0
		self.features['fans_0_50'][(self.features.count_of_fans > 0) & (self.features.count_of_fans <= 50)] = 1

		self.features['fans_50_100'] = 0
		self.features['fans_50_100'][(self.features.count_of_fans > 50) & (self.features.count_of_fans <= 100)] = 1

		self.features['fans_100_200'] = 0
		self.features['fans_100_200'][(self.features.count_of_fans > 100) & (self.features.count_of_fans <= 200)] = 1

		self.features['fans_200_500'] = 0
		self.features['fans_200_500'][(self.features.count_of_fans > 200) & (self.features.count_of_fans <= 500)] = 1

		self.features['fans_500_1000'] = 0
		self.features['fans_500_1000'][(self.features.count_of_fans > 500) & (self.features.count_of_fans <= 1000)] = 1

		self.features['fans_1000'] = 0
		self.features['fans_1000'][(self.features.count_of_fans > 1000)] = 1