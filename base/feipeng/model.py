import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression


class age(object):
	def __init__(self, data_file, feature_file, stack_file, train_labels):
		#保存stack特征
		self.stack_file = stack_file
		# 特征和微博数据X
		self.features = pickle.load(open(feature_file, 'rb'))
		self.data_x = pickle.load(open(data_file, 'rb'))

		# 标签Y
		self.df = pd.read_csv(train_labels, names=['uid','gender','birthday','location'], encoding='utf-8')
		def age(x):
		    if x <= 1979:
		        return u'-1979'
		    elif x>=1980 and x<=1989:
		        return u'1980-1989'
		    else:
		        return u'1990+'
		self.df['age'] = self.df.birthday.apply(lambda x:age(x))

		self.df = self.df[['uid','age']]

		self.le_age = LabelEncoder()
		self.le_age.fit(self.df.age)
		self.df['y_age'] = self.le_age.transform(self.df.age)

	def stacking(self):
		X = self.data_x.weibo_and_source[:]
		vectormodel = TfidfVectorizer(ngram_range=(1,1), min_df=3,use_idf=False, smooth_idf=False, sublinear_tf=True, norm=False)
		X = vectormodel.fit_transform(X)

		# 数据
		y = self.df.y_age
		train_x = X[:len(y)]
		test_x = X[len(y):].tocsc()

		np.random.seed(0)

		n_folds = 5
		n_class = 3

		train_x_id = range(train_x.shape[0])
		val_x_id = range(test_x.shape[0])

		X = train_x
		y = y
		X_submission = test_x

		    
		skf = list(StratifiedKFold(y, n_folds))

		clfs = [
		    LogisticRegression(penalty='l1',n_jobs=-1,C=1.0),
		    LogisticRegression(penalty='l2',n_jobs=-1,C=1.0),
		    RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
		    RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),  
		    ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),  
		    ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy')
		]


		dataset_blend_train = np.zeros((X.shape[0], len(clfs)*n_class))
		dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)*n_class))

		for j, clf in enumerate(clfs):
			print (j, clf)
			dataset_blend_test_j = np.zeros((X_submission.shape[0], n_class))
			for i, (train, test) in enumerate(skf):
				print ('Fold ',i)
				X_train = X[train]
				y_train = y[train]
				X_test = X[test]
				y_test = y[test]
				clf.fit(X_train, y_train)
				y_submission = clf.predict_proba(X_test)
				dataset_blend_train[test, j*n_class:j*n_class+n_class] = y_submission
				dataset_blend_test_j += clf.predict_proba(X_submission)
			dataset_blend_test[:,j*n_class:j*n_class+n_class] = dataset_blend_test_j[:,]/n_folds

		all_X_1 = np.concatenate((dataset_blend_train, dataset_blend_test), axis=0)

		# xgboost
		temp = np.zeros((len(y),n_class))
		test = np.zeros((test_x.shape[0], n_class))
		test_x = test_x.tocsc()
		dtest = xgb.DMatrix(test_x)
		for tra, val in StratifiedKFold(y, 5, random_state=658):
		    X_train = train_x[tra]
		    y_train = y[tra]
		    X_val = train_x[val]
		    y_val = y[val]
		    
		    x_train = X_train.tocsc()
		    x_val = X_val.tocsc()
		    
		    dtrain = xgb.DMatrix(x_train, y_train)
		    dval = xgb.DMatrix(x_val)
		    
		    params = {
		    "objective": "multi:softprob",
		    "booster": "gblinear",
		    "eval_metric": "merror",
		    "num_class":3,
		    'max_depth':3,
		    'min_child_weight':1.5,
		    'subsample':0.7,
		    'colsample_bytree':1,
		    'gamma':2.5,
		    "eta": 0.01,
		    "lambda":1,
		    'alpha':0,
		    "silent": 1,
		    }
		    watchlist = [(dtrain, 'train')]
		    model = xgb.train(params, dtrain, 2000, evals=watchlist,
		                early_stopping_rounds=200, verbose_eval=200)
		    result = model.predict(dval)
		    temp[val] = result[:]
		    
		    res = model.predict(dtest)
		    test += res
		test /= 5
		all_X_2 = np.concatenate((temp, test), axis=0)

		#############################################################################
		#############################################################################
		# merge
		all_X = np.concatenate((all_X_1, all_X_2), axis=1)
		pickle.dump(all_X, open(self.stack_file,'wb'))

	def concat_features(self, outfeatures=None):
		print('concat features...')
		all_X = pickle.load(open(self.stack_file,'rb'))
		myfeature = self.features.drop(['uid'],axis=1).as_matrix()     
		#train+test set
		self.all_X = np.concatenate((all_X, myfeature), axis=1)
		
		#concat_outfea
		if outfeatures:
			featureslist = pickle.load(open(outfeatures, 'rb'))
			for fea in featureslist:
				self.all_X = np.concatenate((self.all_X, fea), axis=1)
		#train set
		self.X = self.all_X[:self.df.shape[0]]
		self.y = self.df.y_age

		print ('特征维数为{}维'.format(self.X.shape[1]))

	def fit_transform(self, result_age):
		print('bagging...')
		n = 8
		score = 0
		pres = []
		i=1
		for tra, val in StratifiedShuffleSplit(self.y, n, test_size=0.2, random_state=233):
		    print('run {}/{}'.format(i,n))
		    i+=1
		    
		    X_train = self.X[tra]
		    y_train = self.y[tra]
		    X_val = self.X[val]
		    y_val = self.y[val]
		    
		    dtrain = xgb.DMatrix(X_train, y_train)
		    dval = xgb.DMatrix(X_val, y_val)
		    dtest = xgb.DMatrix(self.all_X[self.df.shape[0]:])

		    params = {
		        "objective": "multi:softmax",
		        "booster": "gbtree",
		        "eval_metric": "merror",
		        "num_class":3,
		        'max_depth':3,
		        'min_child_weight':1.5,
		        'subsample':0.7,
		        'colsample_bytree':1,
		        'gamma':2.5,
		        "eta": 0.01,
		        "lambda":1,
		        'alpha':0,
		        "silent": 1,
		    }
		    watchlist = [(dtrain, 'train'), (dval, 'eval')]

		    bst = xgb.train(params, dtrain, 2000, evals=watchlist,
		                    early_stopping_rounds=200, verbose_eval=False)
		    score += bst.best_score

		    pre = bst.predict(dtest)
		    pres.append(pre)
		    
		score /= n
		score = 1 - score
		print('*********************************************')
		print('*********************************************')
		print("******年龄平均准确率为{}**************".format(score))
		print('*********************************************')
		print('*********************************************')

		# vote
		pres = np.array(pres).T.astype('int64')
		pre = []
		for line in pres:
		    pre.append(np.bincount(line).argmax())

		result = pd.DataFrame(pre, columns=['age'])
		result['age'] = result.age.apply(lambda x: int(x))
		result['age'] = self.le_age.inverse_transform(result.age)
		result.to_csv(result_age, index=None)
		print('result saved!')





class gender(object):
	def __init__(self, data_file, feature_file, stack_file, train_labels):
		#保存stack特征
		self.stack_file = stack_file
		# 特征和微博数据X
		self.features = pickle.load(open(feature_file, 'rb'))
		self.data_x = pickle.load(open(data_file, 'rb'))

		# 标签Y
		self.df = pd.read_csv(train_labels, names=['uid','gender','birthday','location'], encoding='utf-8')
		self.df = self.df[['uid','gender']]

		self.le_gender = LabelEncoder()
		self.le_gender.fit(self.df.gender)
		self.df['y_gender'] = self.le_gender.transform(self.df.gender)

	def stacking(self):
		X = self.data_x.weibo_and_source[:]
		vectormodel = TfidfVectorizer(ngram_range=(1,1), min_df=3,use_idf=False, smooth_idf=False, sublinear_tf=True, norm=False)
		X = vectormodel.fit_transform(X)

		# 数据
		y = self.df.y_gender
		train_x = X[:len(y)]
		test_x = X[len(y):].tocsc()

		np.random.seed(9)

		n_folds = 5
		n_class = 2

		train_x_id = range(train_x.shape[0])
		val_x_id = range(test_x.shape[0])

		X = train_x
		y = y
		X_submission = test_x


		skf = list(StratifiedKFold(y, n_folds, random_state=99))

		clfs = [
		        LogisticRegression(penalty='l1',n_jobs=-1,C=1.0),
		        LogisticRegression(penalty='l2',n_jobs=-1,C=1.0),
		        RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='gini', random_state=9),
		        RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='entropy', random_state=9),  
		        ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='gini', random_state=9),  
		        ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='entropy', random_state=9)
		]


		dataset_blend_train = np.zeros((X.shape[0], len(clfs)*n_class))
		dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)*n_class))

		for j, clf in enumerate(clfs):
			np.random.seed(9)
			print (j, clf)
			dataset_blend_test_j = np.zeros((X_submission.shape[0], n_class))
			for i, (train, test) in enumerate(skf):
				print ('Fold ',i)
				X_train = X[train]
				y_train = y[train]
				X_test = X[test]
				y_test = y[test]
				clf.fit(X_train, y_train)
				y_submission = clf.predict_proba(X_test)
				dataset_blend_train[test, j*n_class:j*n_class+n_class] = y_submission
				dataset_blend_test_j += clf.predict_proba(X_submission)
			dataset_blend_test[:,j*n_class:j*n_class+n_class] = dataset_blend_test_j[:,]/n_folds

		all_X_1 = np.concatenate((dataset_blend_train, dataset_blend_test), axis=0)

		# xgboost
		temp = np.zeros((len(y),n_class))
		test = np.zeros((test_x.shape[0], n_class))
		test_x = test_x.tocsc()
		dtest = xgb.DMatrix(test_x)
		for tra, val in StratifiedKFold(y, 5, random_state=23):
		    X_train = train_x[tra]
		    y_train = y[tra]
		    X_val = train_x[val]
		    y_val = y[val]
		    
		    x_train = X_train.tocsc()
		    x_val = X_val.tocsc()
		    
		    dtrain = xgb.DMatrix(x_train, y_train)
		    dval = xgb.DMatrix(x_val)
		    
		    params = {
		    "objective": "multi:softprob",
		    "booster": "gblinear",
		    "eval_metric": "merror",
		    "num_class":2,
		    'max_depth':3,
		    'min_child_weight':1.5,
		    'subsample':0.7,
		    'colsample_bytree':1,
		    'gamma':2.5,
		    "eta": 0.01,
		    "lambda":1,
		    'alpha':0,
		    "silent": 1,
		    'seed':1
		    }
		    watchlist = [(dtrain, 'train')]
		    model = xgb.train(params, dtrain, 2000, evals=watchlist,
		                early_stopping_rounds=50, verbose_eval=1000)
		    result = model.predict(dval)
		    temp[val] = result[:]
		    
		    res = model.predict(dtest)
		    test += res
		test /= n_folds
		all_X_2 = np.concatenate((temp, test), axis=0)

		###############################
		###############################
		all_X = np.concatenate((all_X_1, all_X_2), axis=1)
		pickle.dump(all_X, open(self.stack_file, 'wb'))

	def fit_transform(self, result_gender):
		print('concat features...')
		all_X = pickle.load(open(self.stack_file,'rb'))
		myfeature = self.features.drop(['uid'],axis=1).as_matrix()     
		#train+test set
		self.all_X = np.concatenate((all_X, myfeature), axis=1)
		#train set
		self.X = self.all_X[:self.df.shape[0]]
		self.y = self.df.y_gender
		print ('特征维数为{}维'.format(self.X.shape[1]))

		print('bagging')
		n = 7
		score = 0
		pres = []
		i=1
		for tra, val in StratifiedShuffleSplit(self.y, n, test_size=0.2, random_state=7):
		    
		    print('run {}/{}'.format(i,n))
		    i+=1
		    
		    X_train = self.X[tra]
		    y_train = self.y[tra]
		    X_val = self.X[val]
		    y_val = self.y[val]
		    
		    dtrain = xgb.DMatrix(X_train, y_train)
		    dval = xgb.DMatrix(X_val, y_val)
		    dtest = xgb.DMatrix(self.all_X[self.df.shape[0]:])

		    params = {
		        "objective": "binary:logistic",
		        "booster": "gbtree",
		        "eval_metric": "error",
		        'max_depth':3,
		        'min_child_weight':1.5,
		        'subsample':0.7,
		        'colsample_bytree':1,
		        'gamma':2.5,
		        "eta": 0.01,
		        "lambda":1,
		        'alpha':0,
		        "silent": 1,
		    }
		    watchlist = [(dtrain, 'train'), (dval, 'eval')]

		    bst = xgb.train(params, dtrain, 2000, evals=watchlist,
		                    early_stopping_rounds=200, verbose_eval=False)
		    score += bst.best_score
		    
		    pre = bst.predict(dtest)
		    pre[pre>=0.5] = 1
		    pre[pre<0.5] = 0
		    pres.append(pre)
		    
		    
		score /= n
		score = 1 - score
		print('*********************************************')
		print('*********************************************')
		print("******性别平均准确率为{}**************".format(score))
		print('*********************************************')
		print('*********************************************')

		# vote
		pres = np.array(pres).T.astype('int64')
		pre = []
		for line in pres:
		    pre.append(np.bincount(line).argmax())

		result = pd.DataFrame(pre, columns=['gender'])
		result['gender'] = self.le_gender.inverse_transform(result.gender)
		result.to_csv(result_gender, index=None)