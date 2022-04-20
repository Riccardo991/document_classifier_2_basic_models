
# Document classifier 2 basic models
# The augmented dataset now has the follow classes distribution (0 = 3984, 1 = 1935, 2 = 550).
# The goal of this script is to implement and test a logistic regression classifier and svm.

import pandas as pd 
import numpy as np 
import time, json, pickle  
from collections import Counter 
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder  
from sklearn.experimental import enable_halving_search_cv    
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV, RepeatedStratifiedKFold 
from sklearn.decomposition import PCA, TruncatedSVD 
from sklearn.linear_model import  LogisticRegression 
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.metrics import accuracy_score, log_loss, f1_score, balanced_accuracy_score  

def splitData ( df ):
    kl = np.unique( np.array(df['Labels']))
    df_training = pd.DataFrame()
    df_test = pd.DataFrame()
    for k in kl:
        df_class = df[ df['Labels'] == k]
        n = int( df_class.shape[0]/10)
        df_tr = df_class[:-n]
        df_ts = df_class[-n:]
        df_training= pd.concat([df_training, df_tr], axis=0)
        df_test = pd.concat([ df_test, df_ts], axis=0)
    for i in range(0, 3):
        df_training= shuffle( df_training, random_state=33)
        df_test = shuffle(df_test, random_state=33)
    return df_training, df_test

print("go")
df = pd.read_excel('...\\df_ramo_corpus_big_with_keywords.xlsx')
print("df size: ",df.shape)
print("true   labels ",Counter(df['Ramo']))

# Transform categorical labels into  numbers
df = df[['corpus', 'Ramo']]
le = LabelEncoder()
labels  = le.fit_transform(df['Ramo'])
df['Labels'] = labels 
df = shuffle( df, random_state=52)
# Split the dataset into training set and test set and mix the rows 
df_training, df_test = splitData( df)
print(" training ",df_training.shape," test ",df_test.shape)
print("labels on test set ",Counter(df_test['Labels']))

# Create the Tf-Idf matrix and apply  the latent semantic analysis  to reduce the features  for logistic reggression 
tv1 = TfidfVectorizer( min_df=10, norm='l2', use_idf=True )
x_set = tv1.fit_transform( df_training['corpus'])
x_set = x_set.toarray()
voc = tv.get_feature_names()
print(" the number of words in tf  idf vocavolari is ",len(voc))
svd1  = TruncatedSVD( n_components=2000, random_state=42)  
x_set = svd1.fit_transform(x_set )
print("  svd reductions ",x_set.shape)
y_set = df_training['Labels'].values 


# Training logistic regression model with the cross validation 
rskf = RepeatedStratifiedKFold( n_repeats=3, n_splits=5, random_state=77) 
lr = LogisticRegression ( class_weight='balanced', max_iter=200)
acList = []
f1List = []
lgList = []

for training_index, validation_index in rskf.split( x_set, y_set):
    x_tr, x_val = x_set[training_index], x_set[validation_index ]
    y_tr, y_val = y_set[training_index], y_set[ validation_index ]
    lr.fit(x_tr, y_tr)
    y_pred = lr.predict(x_val)
    ac = accuracy_score( y_val, y_pred)
    acList.append( ac)
    f1 = f1_score( y_val, y_pred, average='weighted')
    f1List.append( f1 )
    y_prob = lr.predict_proba( x_val)
    lg = log_loss( y_val, y_prob)
    lgList.append( lg )

va, vf, vl = np.array( acList).mean(), np.array( f1List ).mean(), np.array( lgList).mean()
print("LogisticRegression  training: accuracy %.4f, f1 %.4f, loss %.4f " %(va, vf, vl))

# test the model 
x_ts = tv1.transform( df_test['corpus'])
x_ts = x_ts.toarray()
x_ts = svd1.transform( x_ts)
y_ts = df_test['Labels'].values
print(" x_ts ",x_ts.shape," y_ts ",y_ts.shape)
y_pred = lr.predict( x_ts)
ac_lr = accuracy_score(y_ts, y_pred)
f1_lr = f1_score( y_ts, y_pred, average='weighted')
bas_lr = balanced_accuracy_score( y_ts, y_pred)
y_prob = lr.predict_proba( x_ts)
lg_lr = log_loss( y_ts, y_prob)
print(" predicted labels ",Counter(y_pred))
print("LogisticRegression  test: accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_lr, lg_lr, f1_lr, bas_lr))

# Apply the CalibratedClassifierCV   on the model to improve the accuracy on the unbalanced data
cl_lr = CalibratedClassifierCV (lr, method='isotonic', cv=5)
cl_lr.fit( x_set, y_set)
y_pred = cl_lr.predict( x_ts)
ac_cl = accuracy_score(y_ts, y_pred)
f1_cl = f1_score( y_ts, y_pred, average='weighted')
bas_cl = balanced_accuracy_score( y_ts, y_pred)
y_prob = cl_lr.predict_proba( x_ts)
lg_cl = log_loss( y_ts, y_prob)
print(" predicted labels ",Counter(y_pred))
print("CalibratedClassifierCV test: accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_cl, lg_cl, f1_cl, bas_cl))

# I create the Tf-Idf matrix and apply  the latent semantic analysis  to reduce the features  for SVM
tv2 = TfidfVectorizer( min_df=0.02, norm='l2', max_df=0.98, use_idf=True )
x_set = tv2.fit_transform( df_training['corpus'])
x_set = x_set.toarray()
voc = tv.get_feature_names()
print(" the number of words in tf  idf vocavolari is ",len(voc))
svd2  = TruncatedSVD( n_components=900, random_state=42)  
x_set = svd2.fit_transform(x_set )
print("  svd reductions ",x_set.shape)
y_set = df_training['Labels'].values 

# Training the svm model using the GridSearch to find the model with the best parameters
svm = SVC(probability= True, class_weight='balanced' )
svm_params = { 'kernel':['linear', 'rbf', 'sigmoid'],
              'C':[1, 10, 20],
          'gamma': ['auto', 1, 0.1 ]}
rcv = StratifiedKFold( n_splits=5, shuffle=True, random_state= 42 )

svm_model = HalvingGridSearchCV ( svm, svm_params,  cv=rcv, factor=3, min_resources='exhaust', aggressive_elimination= False,  scoring="f1_weighted", n_jobs=-1, verbose=1)                                 
svm_model.fit(x_set, y_set)
best_parameters = svm_model.best_params_
print(" the best params are:: \n ",best_parameters)
# test the best model 
x_ts = tv2.transform( df_test['corpus'])
x_ts = x_ts.toarray()
x_ts = svd2.transform( x_ts)
y_ts = df_test['Labels'].values
print(" x_ts ",x_ts.shape," y_ts ",y_ts.shape)
svm = svm_model.best_estimator_
y_pr = svm.predict( x_ts)
ac_svm = accuracy_score( y_ts, y_pr)
f1_svm = f1_score( y_ts, y_pr, average='weighted')
bas_svm = balanced_accuracy_score  (y_ts, y_pr)
y_pb = svm.predict_proba( x_ts)
lg_svm = log_loss( y_ts, y_pb)
print(" svm test: accuratezza %.4f, loss %.4f, f1 %.4f, bca %.4f " %(ac_svm, lg_svm, f1_svm, bas_svm))

# save the models 
tv1_model = 'tv1_transform.sav'
pickle.dump( tv1, open( tv1_model, 'wb'))

lr_model = 'lr_model.sav'
pickle.dump( cl_lr, open( lr_model, 'wb'))

tv2_model = 'tv2_transform.sav'
pickle.dump( tv2, open( tv2_model, 'wb'))

svm_model = 'svm_model.sav'
pickle.dump( svm, open( svm_model, 'wb'))

#  write the results in a txt 
with open('...\\results.txt', 'a') as f:
    f.write('\n results  \n')
    w1 = "\n LogisticRegression  test: accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_lr, lg_lr, f1_lr, bas_lr)
    f.write( w1)
    w2 = "\n CalibratedClassifierCV test: accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_cl, lg_cl, f1_cl, bas_cl)
    f.write( w2 )
    f.write(" \n SVM \n best params: \n")
    f.write( json.dumps(best_parameters))
    w3 = "\n svm test: accuratezza %.4f, loss %.4f, f1 %.4f, bca %.4f " %(ac_svm, lg_svm, f1_svm, bas_svm)
    f.write(w3 )

print("end")
