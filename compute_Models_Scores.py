import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import  VotingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
import os
import os.path 
import warnings
import sys
import json
from sklearn.feature_selection import SelectKBest, chi2,mutual_info_classif,f_classif,f_regression,mutual_info_regression
from copy import copy

warnings.filterwarnings("ignore")


def class_accuracy(y_pred, y_true, cl):
    y_pred_ = np.array([int(cl==y) for y in y_pred])
    y_true_ = np.array([int(cl==y) for y in y_true])
    return balanced_accuracy_score(y_true_, y_pred_)

def computeScores(y_true, y_pred,target,flag,target_labels):
    o = {}
    if flag==0:
        for j,l in enumerate(set(y_true)):
            name = target_labels[target][j]
            y_true_l = np.array([int(y==l) for y in y_true])
            y_pred_l = np.array([int(y==l) for y in y_pred])           
            o['Pr '+name] = precision_score(y_true_l, y_pred_l)
            o['Rc '+name] = recall_score(y_true_l, y_pred_l)
            o['F1 '+name] = f1_score(y_true_l, y_pred_l)
            o["Acc Bal "+name] = class_accuracy(y_pred, y_true, l)
        for v in ['micro','macro','weighted']:
            o['Pr '+v] = precision_score(y_true, y_pred,average=v)
            o['Rc '+v] = recall_score(y_true, y_pred,average=v)
            o['F1 '+v] = f1_score(y_true, y_pred,average=v)
        o["Acc Bal"] = balanced_accuracy_score(y_true, y_pred)
    else:
        o['Mse'] = mean_squared_error(y_pred, y_true)
        o['R2'] = r2_score(y_pred, y_true)
    return o

#reading files

args = sys.argv

if len(args) > 1:
    dataset = args[1]
else:
    raise Exception("You have to specify the dataset")


datasets_folder = "datasets"
specifications_path = "/".join([datasets_folder,dataset,"specifications.json"])
data_path = "/".join([datasets_folder,dataset,"data.csv"])
data_path_train = "/".join([datasets_folder,dataset,"data_test.csv"])
data_path_test = "/".join([datasets_folder,dataset,"data_train.csv"])
target_path = "/".join([datasets_folder,dataset,"target.csv"])
target_path_train = "/".join([datasets_folder,dataset,"target_train.csv"])
target_path_test = "/".join([datasets_folder,dataset,"target_test.csv"])

try:
    specifications = json.load(open(specifications_path,"r"))
except:
    specifications = {}
    print("No specifications file selected, the default settings were used")

flg = True

try:
    DF = pd.read_csv(data_path)
    flg = True
except:
    try:
        flg = False
        DF = pd.read_csv(target_path_train)
        DF_TEST = pd.read_csv(target_path_test)
    except:
        raise Exception("Data file not founded")


try:
    if flg:
        TARGET = pd.read_csv(target_path)
    else:
        TARGET = pd.read_csv(target_path_train)
        TARGET_TEST = pd.read_csv(target_path_test)
except:
    raise Exception("Target file not founded")

#feature families creation

if 'ID' not in DF.columns:
    ids = [str(c) for c in range(len(DF))]
    DF['ID'] = ids
    TARGET['ID'] = ids
    if not flg:
        DF_TEST['ID'] = ids
        TARGET_TEST['ID'] = pd.read_csv(target_path_test)


features_families = {"All":[f for f in list(DF.columns) if f != 'ID']}

if "features" in specifications and "families" in specifications["features"]:
    for f in specifications["features"]['families']:
        fs = specifications["features"]['families'][f]
        for f_ in fs:
            if f_ not in features_families['All']:
                del specifications["features"]['families'][f]
                raise Exception(f_+" feature is not present in the dataset")
        features_families[f] = specifications["features"]['families'][f]

#target labels


target_algorithms = dict()
if "target" in specifications and "type" in specifications["target"]:
    target_algorithms = specifications["target"]['type']


targets = list(TARGET)
for t in targets:
    if t not in target_algorithms:
        target_algorithms[t] = ['classification','regression']

target_labels = {
    t:[str(v) for v in set(TARGET[t])]
    for t in targets
}


if "target" in specifications and "labels" in specifications["target"]:
    for t in specifications["target"]['labels']:
        target_labels[t] = specifications["target"]['labels'][t]


#algorithms creation

all_scores = list()

from datetime import datetime
classifiers = [
    [KNeighborsClassifier(),[{'n_neighbors':[2,3,4,5,6,7,8,9]}]],
    [SVC(),[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.025,0.05,0.01,1, 10, 100, 1000]}]],
    [DecisionTreeClassifier(),[{'max_depth': [3,5,7,10,None], 'min_samples_split': [2,4,6],
                     'min_samples_leaf': [1,3,5],"max_features":[None,'auto','sqrt','log2']}]],
    [RandomForestClassifier(),[{'max_depth': [3,5,7,10,None], 'min_samples_split': [2,4,6],
                     'min_samples_leaf': [1,3,5],"max_features":[None,'auto','sqrt','log2'],"n_estimators":[10,50,100]}]],
    GaussianNB(),GaussianProcessClassifier(1.0 * RBF(1.0)),MLPClassifier(),LogisticRegression()
]


regressors = [
    [KNeighborsRegressor(),[{'n_neighbors':[2,3,4,5,6,7,8,9]}]],
    [SVR(),[{'kernel': ['rbf'],'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [0.025,0.05,0.01,1, 10, 100, 1000]}]],
    [DecisionTreeRegressor(),[{'max_depth': [3,5,7,10,None], 'min_samples_split': [2,4,6],
                     'min_samples_leaf': [1,3,5],"max_features":[None,'auto','sqrt','log2']}]],
    [RandomForestRegressor(),[{'max_depth': [3,5,7,10,None], 'min_samples_split': [2,4,6],
                     'min_samples_leaf': [1,3,5],"max_features":[None,'auto','sqrt','log2'],"n_estimators":[10,50,100]}]],
    GaussianProcessRegressor(1.0 * RBF(1.0)),MLPRegressor(),LinearRegression()
]


try:
    os.makedirs("files")
except:
    pass


def formData(X,y,flg,cols,data_path,n_feats,sel):
    kf = StratifiedShuffleSplit(n_splits=5,random_state=3,test_size=0.15)
    rs = ShuffleSplit(n_splits=5,random_state=3,test_size=0.15)
    if flg:
        try:
            spl = list(kf.split(X,y))
            fl = True
        except:
            spl = list(rs.split(X))
            fl = False
        data_objs = []
        for i,(train_index, test_index) in enumerate(spl):
            print('Split:',i)
            X_train_, X_test = X[train_index], X[test_index]
            y_train_, y_test = y[train_index], y[test_index]
            if sel != -1:
                X_train_,X_test,ncols= reduceDimension(X_train_, X_test, y_train_, y_test,n_feats,sel,copy(cols))
                data_obj = {
                    "train":{
                        "X":pd.DataFrame(X_train_,columns=ncols),
                        'Y':y_train_
                    },
                    "test":{
                        "X":pd.DataFrame(X_test,columns=ncols),
                        'Y':y_test
                    }
                }
            else:
                data_obj = {
                    "train":{
                        "X":pd.DataFrame(X_train_,columns=cols),
                        'Y':y_train_
                    },
                    "test":{
                        "X":pd.DataFrame(X_test,columns=cols),
                        'Y':y_test
                    }
                }
            DP = data_path + "/"+str(i)
            try:
                os.makedirs(DP)
            except:
                pass
            pickle.dump(data_obj,open(DP+'/data.p','wb'))
            data_objs.append(data_obj)    
    else:
        X_ = X[0]
        y_ = Y[0]
        data_obj = {
            "train":{
                "X":pd.DataFrame(X[0],columns=DATA.columns),
                'Y':Y[0]
            },
            "test":{
                "X":pd.DataFrame(X[1],columns=DATA.columns),
                'Y':Y[1]
            }
        }
        DP = data_path + "/"+str(i)
        try:
            os.makedirs(DP)
        except:
            pass
        pickle.dump(data_obj,open(DP+'/data.p','wb'))
        data_objs= [data_obj]
    return data_objs

targets = [t for t in targets if t != 'ID']

selection_methods = [chi2,mutual_info_classif,f_classif,f_regression,mutual_info_regression]


for t in targets:
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    print('Time:',timestampStr)
    print('Target:',t)
    for j,feats in enumerate(features_families):
        print('Features:',feats,j)
        valid_ids = list(TARGET[TARGET[t]!=-1]['ID'])
        features=features_families[feats]
        DF_ = DF[DF['ID'].isin(valid_ids)]
        DATA=DF_[features]
        X = DATA.values
        y = np.array(TARGET[TARGET[t]!=-1][t])

        data_path = "/".join(['files',dataset,'data',feats,t])
        data_objs = formData(X,y,flg,features,data_path,0,-1)

        for i,data_obj in enumerate(data_objs):
            X_train = data_obj['train']['X'].values
            y_train = data_obj['train']['Y']
            X_test = data_obj['test']['X'].values
            y_test = data_obj['test']['Y']
            mods = {}

            for typ_alg in target_algorithms[t]:
                if 'classification' == typ_alg:
                    models = classifiers
                    flag = 0
                else:
                    models = regressors
                    flag = 1
                for c in models:
                    try:
                        classifier_name = str(c[0]).split('(')[0]
                    except:
                        classifier_name = str(c).split('(')[0]
                    print('Model:',classifier_name)
                    D = "/".join(['files',dataset,'models',feats,t,str(i),classifier_name,"-1"])
                    try:
                        os.makedirs(D)
                    except:
                        pass 
                    D += '/model.p'
                    try:
                        M = pickle.load(open(D,'rb'))
                    except:
                        print("Not done")
                        if type(c) == list:
                            if flag==0:
                                M = GridSearchCV(
                                    c[0], c[1], scoring=make_scorer(balanced_accuracy_score)
                                )
                                M.fit(X_train,y_train)
                            elif flag==1:
                                M = GridSearchCV(
                                    c[0], c[1], scoring=make_scorer(mean_squared_error)
                                )
                                M.fit(X_train,y_train) 
                            M = M.best_estimator_
                        else:
                            M = c
                        M.fit(X_train,y_train)
                        pickle.dump(M,open(D,'wb'))

                    print("Done:",D)

                    mods[classifier_name]=M

                    y_pred=M.predict(X_test)

                    scores = computeScores(y_test, y_pred,t,flag,target_labels)

                    scores['target'] = t

                    scores['algo'] = classifier_name

                    scores["features_family"] = feats

                    scores["set"] = 'Test' 

                    scores["split"] = i 

                    scores["type_alg"] = typ_alg

                    scores["x"] = -1

                    scores["selection_methods"] = str(-1)

                    scores["featsel"] = str(-1)

                    all_scores.append(scores)


                    y_pred=M.predict(X_train)

                    scores = computeScores(y_train, y_pred,t,flag,target_labels)

                    scores['target'] = t

                    scores['algo'] = classifier_name

                    scores["features_family"] = feats

                    scores["set"] = 'Train'

                    scores["split"] = i

                    scores["type_alg"] = typ_alg

                    scores["x"] = -1

                    scores["selection_methods"] = str(-1)

                    scores["featsel"] = str(-1)
                    
                    all_scores.append(scores)

                    scores_path = "/".join(['files',dataset,'all_scores.p'])

                    print("scorespath",scores_path)

                    pickle.dump(all_scores,open(scores_path,'wb'))

        for sel in selection_methods:

            for x in range(5,len(features)-1,5):


                data_path = "/".join(['files',dataset,'data',feats,t])
                data_objs = formData(X,y,flg,features,data_path,x,sel)

                for i,data_obj in enumerate(data_objs):

                    featsel = list(data_obj['train']['X'].columns)

                    X_train = data_obj['train']['X'].values
                    y_train = data_obj['train']['Y']
                    X_test = data_obj['test']['X'].values
                    y_test = data_obj['test']['Y']

                    mods = {}

                    for typ_alg in target_algorithms[t]:
                        if 'classification' == typ_alg:
                            models = classifiers
                            flag = 0
                        else:
                            models = regressors
                            flag = 1
                        for c in models:
                            try:
                                classifier_name = str(c[0]).split('(')[0]
                            except:
                                classifier_name = str(c).split('(')[0]
                            print('Model:',classifier_name)
                            D = "/".join(['files',dataset,'models',feats,t,str(i),classifier_name,sel.__name__+'_'+str(x)])
                            try:
                                os.makedirs(D)
                            except:
                                pass 
                            D += '/model.p'
                            try:
                                M = pickle.load(open(D,'rb'))
                            except:
                                print("Not done")
                                if type(c) == list:
                                    if flag==0:
                                        M = GridSearchCV(
                                            c[0], c[1], scoring=make_scorer(balanced_accuracy_score)
                                        )
                                        M.fit(X_train,y_train)
                                    elif flag==1:
                                        M = GridSearchCV(
                                            c[0], c[1], scoring=make_scorer(mean_squared_error)
                                        )
                                        M.fit(X_train,y_train) 
                                    M = M.best_estimator_
                                else:
                                    M = c
                                M.fit(X_train,y_train)
                                pickle.dump(M,open(D,'wb'))

                            print("Done:",D)

                            mods[classifier_name]=M

                            y_pred=M.predict(X_test)

                            scores = computeScores(y_test, y_pred,t,flag,target_labels)

                            scores['target'] = t

                            scores['algo'] = classifier_name

                            scores["features_family"] = feats

                            scores["set"] = 'Test' 

                            scores["split"] = i 

                            scores["type_alg"] = typ_alg

                            scores["x"] = x

                            scores["selection_methods"] = sel.__name__

                            scores["featsel"] = featsel

                            all_scores.append(scores)


                            y_pred=M.predict(X_train)

                            scores = computeScores(y_train, y_pred,t,flag,target_labels)

                            scores['target'] = t

                            scores['algo'] = classifier_name

                            scores["features_family"] = feats

                            scores["set"] = 'Train'

                            scores["split"] = i

                            scores["type_alg"] = typ_alg

                            scores["x"] = x

                            scores["selection_methods"] = sel.__name__

                            scores["featsel"] = featsel
                            
                            all_scores.append(scores)

                            scores_path = "/".join(['files',dataset,'all_scores.p'])

                            print("scorespath",scores_path)

                            pickle.dump(all_scores,open(scores_path,'wb'))

scores_path = "/".join(['files',dataset,'all_scores.p'])
pickle.dump(all_scores,open(scores_path,'wb'))

scores_path_csv = "/".join(['files',dataset,'all_scores.csv'])
pd.DataFrame(all_scores).to_csv(scores_path_csv,index=False)
