import os
from base.feipeng.feature_engineering import features
from base.feipeng.model import age, gender
import pandas as pd

# 输入文件路径
inpaths = {
	'train_info' : 'data/raw_data/train/train_info.txt',
	'train_labels' : 'data/raw_data/train/train_labels.txt',
	'train_links' : 'data/raw_data/train/train_links.txt',
	'train_status' : 'data/raw_data/train/train_status.txt',
	'new_labels' : 'data/raw_data/train/new_labels.txt',

	'test_info' : 'data/raw_data/valid/valid_info.txt',
	'test_nolabels' : 'data/raw_data/valid/valid_nolabel.txt',
	'test_links' : 'data/raw_data/valid/valid_links.txt',
	'test_status' : 'data/raw_data/valid/valid_status.txt',

	'stopwords' : 'data/user_data/stopwords.txt'
}

# 输出文件路径
outpaths = {
	# 中间文件
	'status_file' : 'data/feature_data/fp.status.txt',
	'text_file' : 'data/feature_data/fp.text.txt',
	# 最终文件
	'data_file' : 'data/feature_data/fp.data_x.pkl',
	'features_file' : 'data/feature_data/fp.features.pkl'
}

# stack特征输出文件
stack_age = 'data/models/fp.age.feature'
stack_gender = 'data/models/fp.gender.feature'

# 预测结果
result_age = 'submission/age.csv'
result_gender = 'submission/gender.csv'




####################################################
###############        #############################
##############      ################################
##############        ##############################
####################################################


def preprocess():
	print('##################特征工程#####################')
	if os.path.isfile(outpaths['features_file']):
		print('yo 特征已有！')
	else:
		print('重新构建特征')
		myfeatures = features(inpaths, outpaths)
		myfeatures.build()

def predict_age(lee_features):
	print('##################  age  #######################')
	model_age = age(outpaths['data_file'], outpaths['features_file'],\
					stack_age, inpaths['new_labels'])
	if os.path.isfile(stack_age):
		print('stacked~ 开始训练第二级模型')
	else:
		print('stacking')
		model_age.stacking()
	model_age.concat_features(lee_features)
	model_age.fit_transform(result_age)

def predict_gender():
	print('################## gender  #######################')
	model_gender = gender(outpaths['data_file'], outpaths['features_file'],\
					stack_gender, inpaths['new_labels'])
	if os.path.isfile(stack_gender):
		print('stacked~ 开始训练第二级模型')
	else:
		print('stacking')
		model_gender.stacking()
	model_gender.fit_transform(result_gender)



if __name__ == '__main__':
	###############################################
	##################
	# step1: 预处理
	preprocess()

	###############################################
	##################
	# step2: age
	lee_features = 'data/models/yuml.age.feature'      # from lym
	predict_age(lee_features)

	###############################################
	##################
	# step3: gender
	predict_gender()

	# step4: merge
	print('merge result')
	result = pd.read_csv('submission/temp.csv')
	age_file = pd.read_csv(result_age)
	gender_file = pd.read_csv(result_gender)
	result.age = age_file.age
	result.gender = gender_file.gender
	result.to_csv('submission/final.csv', index=0)

	print('finish!')