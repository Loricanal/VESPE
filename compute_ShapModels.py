import pickle
import pandas as pd
import os
import os.path 
import shap
import warnings
import numpy as np
from types import MethodType
import sys
import argparse
import json

warnings.filterwarnings("ignore")

def predictSingleClass(self,X):
    PR = self.predict(X)
    VAL =  np.array([int(v==self.cl) for v in list(PR)])
    return VAL

def computeShapModels(M,X,T,D,typ_alg,target_labels,maxsamples):
    P = M.predict(X)
    set_p = set(P)
    #input(set_p)
    #input(typ_alg)
    if len(set_p) > 1:
        #print(111)
        if typ_alg == "classification":
            for cl in set(P):
                #print(cl)
                #print(target_labels)
                D_ = D + '/shap_data_'+target_labels[int(cl)]+'.p'
                #print(D_)
                if not os.path.exists(D_):
                    print("Computing class",cl)
                    PS = pd.Series(P)
                    PS.index = list(X.index)
                    if maxsamples:
                        maxsamples_ = int(maxsamples/2)
                        X_1 = X[PS==cl]
                        X_1 = X[PS==cl].sample(n=maxsamples_)
                        X_0 = X[~(PS==cl)].sample(n=maxsamples_)
                        X_01 = pd.concat([X_0,X_1]).sample(frac=1)
                    else:
                        X_01 = X
                    M.predictSingleClass = MethodType(predictSingleClass, M)
                    M.cl = cl
                    explainer = shap.KernelExplainer(M.predictSingleClass,X_01)
                    shap_values = explainer.shap_values(X_01)
                    o = {"expected_value":explainer.expected_value,"shap_values":shap_values,"X":X_01}
                    print("Done1:",D_)
                    pickle.dump(o,open(D_,'wb'))
                else:
                    print("Data already computed")
        else:
            print("Computing grade")
            if maxsamples:
                X_ = X.sample(maxsamples)
            X_ = X
            D_ = D + '/shap_data.p'
            explainer = shap.KernelExplainer(M.predict,X_)
            shap_values = explainer.shap_values(X_)
            o = {"expected_value":explainer.expected_value,"shap_values":shap_values,"X":X_}
            pickle.dump(o,open(D_,'wb'))


args = sys.argv
if len(args) > 1:
    dataset = args[1]
else:
    raise Exception("You have to specify the dataset")

datasets_folder = "datasets"
specifications_path = "/".join([datasets_folder,dataset,"specifications.json"])
try:
    specifications = json.load(open(specifications_path,"r"))
except:
    specifications = {}
    print("No specifications file selected, the default settings were used")

target_path = "/".join([datasets_folder,dataset,"target.csv"])
try:
    TARGET = pd.read_csv(target_path)
except:
    raise Exception("Target file not founded")

args=sys.argv[2:]
parser = argparse.ArgumentParser(description="Parses command")
parser.add_argument("-l", "--limit", help="Limit of models to compute")
parser.add_argument("-m", "--maxsamples", help="Limit of models to compute")

options = parser.parse_args(args)

if options.limit:
    limit = int(options.limit)
else:
    limit = ""

if options.maxsamples:
    maxsamples = int(options.maxsamples)
else:
    maxsamples = None


scores_path = "/".join(['files',dataset,'all_scores.p'])
all_scores = pickle.load(open(scores_path,'rb'))
SCORES = pd.DataFrame(all_scores)
fams = set(SCORES['features_family'])
targets = set(SCORES['target'])



target_algorithms = dict()
if "target" in specifications and "type" in specifications["target"]:
    target_algorithms = specifications["target"]['type']
for t in targets:
    if t not in target_algorithms:
        target_algorithms[t] = ['classification','regression']


target_labels = {
    t:{v:str(v) for v in set(TARGET[t]) if v != -1 and not (t=="Grade Sess 1" and v == 0)}
    for t in targets
}





#input(target_labels)



if "target" in specifications and "labels" in specifications["target"]:
    #input(specifications["target"]['labels'])
    for t in specifications["target"]['labels']:
        values = set(TARGET[t])
        if "Grade" in t:
            values = values - {0,1}
        target_labels[t] = {v:specifications["target"]['labels'][t][j] for j,v in enumerate(values)}


target_labels["Grade Sess 1"] = {1: 'Low', 2: 'Medium', 3: 'High'}
print("Limit",limit)



#fams = ['attempts']
for t in targets:
    for type_alg in target_algorithms[t]:
        for f in fams:
            SCORES_sample = SCORES[
                (SCORES['target']==t)&
                (SCORES['set']=="Test")&
                (SCORES['features_family']==f)&
                (SCORES['type_alg']==type_alg)
            ]
            del SCORES_sample['split']
            SCORES_sample = SCORES_sample.dropna(axis=1)
            try:
                SCORES_sample = SCORES_sample.groupby(['algo']).mean().reset_index().sort_values(by=['Acc Bal', 'algo'],ascending=[False,True])
            except:
                SCORES_sample = SCORES_sample.groupby(['algo']).mean().reset_index().sort_values(by=['Mse', 'algo'],ascending=[False,True])
            if not limit:
                limit = len(SCORES_sample)
            for r in SCORES_sample.to_dict(orient='records')[:limit]:
                D = "/".join(['files',dataset,'models',f,t,str(0),r['algo']])
                M = pickle.load(open(D+'/model.p','rb'))
                D = "/".join(['files',dataset,'data',f,t,str(0)])
                data = pickle.load(open(D+'/data.p','rb'))
                X_test = data['test']['X'] 
                X_train = data['train']['X'] 
                X_val = data['train']['X'] 
                X = pd.concat([X_train,X_val,X_test])
                D = "/".join(['files',dataset,'explainers',f,t,str(0),r['algo']])
                if not os.path.isdir(D):
                    os.makedirs(D)
                print("Computing:",D)
                if type_alg == "classification":
                    computeShapModels(M,X,t,D,type_alg,target_labels[t],maxsamples)
                else:
                    computeShapModels(M,X,t,D,type_alg,[],maxsamples)
