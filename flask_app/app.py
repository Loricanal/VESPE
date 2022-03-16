import pandas as pd
import io
import base64
import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import os
from copy import deepcopy
import sys

args = sys.argv

if len(args) > 1:
    dataset = args[1]
else:
    raise Exception("You have to specify the dataset")


def ABS_SHAP(df_shap,df):
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    #input(k2)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value \n(Red = Positive Impact,Blue=Negative Impact)")
    return ax


def cleanFiltObj(filt_obj):
    for key in list(filt_obj.keys()):
        if filt_obj[key] == 'Choose...':
            del filt_obj[key]
        elif key in ['features_family','target','split']:
            filt_obj[key] = filt_obj[key]
    return filt_obj


def formatVals(scores_filt_df):
    num_cols = list()
    oth_cols = list()
    nrs = list()
    for s in scores_filt_df.to_dict(orient='records'):
        ns = dict()
        for key in s:
            if type(s[key])==float:
                ns[key] = '{0:.2f}'.format(s[key]) 
                if key not in num_cols:
                    num_cols.append(key)
            else:
                ns[key] = s[key]
                if key not in oth_cols:
                    oth_cols.append(key)
        nrs.append(ns)
    df_filt = pd.DataFrame(nrs)[oth_cols+num_cols]
    return df_filt

def generateScoreDF(scores_df,filt_obj):
    scores_filt_df = scores_df.copy(deep=True)
    for key in set(filt_obj.keys())-{'class'}:
        scores_filt_df = scores_filt_df[scores_filt_df[key]==filt_obj[key]]
        if key != 'algo':
            del scores_filt_df[key]
    if 'split' in list(scores_filt_df.columns):
        del scores_filt_df['split']
        scores_filt_df = scores_filt_df.groupby(['algo']).mean().reset_index().sort_values(by=['Acc Bal', 'algo'])
    if 'algo' in filt_obj:
        del scores_filt_df['algo']
    scores_filt_df = scores_filt_df.dropna(axis=1)
    return formatVals(scores_filt_df)


def filtScoresByExplainer():
    scores_path = "/".join(['../files',dataset,'all_scores.p'])
    all_scores = pickle.load(open(scores_path,'rb'))
    all_scores = [s_o for s_o in all_scores if 'split' in s_o]
    new_s = list()
    for s in all_scores:
        D = "/".join(
            ['../files',dataset,'explainers',s['features_family'],
            s['target'],"0",s['algo']])
             
        if os.path.isdir(D) and os.listdir(D):
            new_s.append(s)
    return pd.DataFrame(new_s)



datasets_folder = "../datasets"
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

SCORES = filtScoresByExplainer()
features_families = list(set(SCORES['features_family']))
targets = sorted(list(set(SCORES['target'])))
set_types = list(set(SCORES['set']))
splits = list(set(SCORES['split']))
algos = list(set(SCORES['algo']))

target_labels = {
    t:[str(v) for v in set(TARGET[t])]
    for t in targets
}


if "target" in specifications and "labels" in specifications["target"]:
    for t in specifications["target"]['labels']:
        target_labels[t] = specifications["target"]['labels'][t]

classes = list()
for t in targets:
    for c in target_labels[t]:
        if c not in classes:
            classes.append(c)

target_algorithms = dict()
if "target" in specifications and "type" in specifications["target"]:
    target_algorithms = specifications["target"]['type']

for t in targets:
    if t not in target_algorithms:
        target_algorithms[t] = ['classification','regression']


#target_labels
options_filter = list()

for s in SCORES[['features_family','target','algo']].drop_duplicates().to_dict(orient='records'):
    if "classification" in target_algorithms[s['target']]:
        for c in target_labels[s['target']]:
            s1 = deepcopy(s)
            s1['class'] = c
            options_filter.append(s1)
    if "regression" in target_algorithms[s['target']]:
        s['class'] = ''
        options_filter.append(s)



FILT_OBJ = dict()
EXPLAINER = dict()
LABELS_Y = None
Y_SEL = None

def setName(l,target):
    if 'Pass' in target:
        return str(l).replace('0','Fail').replace('1','Pass').replace('2','Not Done')
    else:
        return str(l).replace('0','Ins').replace('1','A').replace('2','B').replace('3','C')




from flask import Flask, render_template
from flask import request, jsonify


app = Flask(__name__)


@app.route("/")
def get_index():
    global features_families,targets,set_types,splits,algos,classes
    return render_template('index.html',
        features_families=features_families,
        targets=targets,
        set_types=set_types,
        splits=splits,
        algos=algos,
        classes=classes
    )

@app.route("/get_options_filter", methods=['GET'])
def get_options_filter():
    global options_filter
    return jsonify(options_filter)


@app.route('/get_data', methods=['POST'])
def get_data_function():
    global SCORES,FILT_OBJ
    filt_obj = request.get_json()
    filt_obj['set'] = "Test"
    FILT_OBJ = filt_obj
    filt_obj = cleanFiltObj(filt_obj)
    scores_filt_df = generateScoreDF(SCORES,filt_obj)
    scores_filt_data = scores_filt_df.to_dict(orient='records')
    return jsonify(scores_filt_data)


@app.route('/loadexplainer',methods=['GET'])
def loadExplainer():
    global EXPLAINER,FILT_OBJ,FEATURES,STUDENTS,LABELS_Y,dataset
    if 'class' in FILT_OBJ and FILT_OBJ['class']:
        D = (
                "/".join(
                ['../files',dataset,'explainers',FILT_OBJ['features_family'],
                FILT_OBJ['target'],"0",FILT_OBJ['algo']])
                 + '/shap_data_'+FILT_OBJ['class']+'.p'
        )
    else:
        D = (
                "/".join(
                ['../files',dataset,'explainers',FILT_OBJ['features_family'],
                FILT_OBJ['target'],"0",FILT_OBJ['algo']])
                 + '/shap_data.p'
        )
    print("Upload dataset",D)
    EXPLAINER = pickle.load(open(D,'rb'))
    if FILT_OBJ['target'] in ['Pass','Pass Sess 0']:
        LABELS_Y = [setName(cl,FILT_OBJ['target']) for cl in [1,2,3]]
    else:
        LABELS_Y = [FILT_OBJ['target']]
    n_classes = len(LABELS_Y)
    FEATURES = list(EXPLAINER['X'].columns)
    STUDENTS = list(EXPLAINER['X'].index)
    #input(STUDENTS)
    return jsonify({'students':STUDENTS,'features':FEATURES})


@app.route('/featureimportance1',methods=['GET'])
def featureimportance1():
    global EXPLAINER,LABELS_Y
    img = io.BytesIO()
    #input(LABELS_Y)
    #input(EXPLAINER["shap_values"])
    #input(EXPLAINER["X"])
    fig = shap.summary_plot(
        EXPLAINER["shap_values"], EXPLAINER["X"], 
        plot_type="bar",show=False,class_names=LABELS_Y
    )
    plt.savefig(img,bbox_inches='tight')
    #plt.savefig("f.png",bbox_inches='tight')
    plt.close()
    img.seek(0)
    my_html = '<img src="data:image/png;base64, {}">'.format(base64.b64encode(img.getvalue()).decode('utf-8'))
    #print(my_html)
    return my_html

@app.route('/featureimportance2',methods=['GET'])
def featureimportance2():
    global EXPLAINER,Y_SEL,LABELS_Y
    img = io.BytesIO()
    fig = shap.summary_plot(
        EXPLAINER["shap_values"], EXPLAINER["X"],show=False
    )
    plt.savefig(img,bbox_inches='tight')
    plt.close()
    img.seek(0)
    my_html = '<img src="data:image/png;base64, {}">'.format(base64.b64encode(img.getvalue()).decode('utf-8'))
    return my_html


@app.route('/featureimportance3',methods=['GET'])
def featureimportance3():
    global EXPLAINER,Y_SEL,LABELS_Y
    img = io.BytesIO()
    ax = ABS_SHAP(EXPLAINER["shap_values"],EXPLAINER["X"])
    plt.rcParams["figure.figsize"] = (20,80)
    ax.figure.savefig(img,bbox_inches='tight')
    plt.savefig(img,bbox_inches='tight')
    plt.close()
    img.seek(0)
    my_html = '<img src="data:image/png;base64, {}">'.format(base64.b64encode(img.getvalue()).decode('utf-8'))
    return my_html

@app.route('/decisionplot',methods=['GET'])
def decisionplot():
    global EXPLAINER,Y_SEL,LABELS_Y
    img = io.BytesIO()


    fig = shap.decision_plot(EXPLAINER["expected_value"],
                            EXPLAINER["shap_values"],
                            EXPLAINER["X"],link='logit',show=False)

    plt.savefig(img,bbox_inches='tight')
    plt.close()
    img.seek(0)
    myhtml = '<img src="data:image/png;base64, {}">'.format(base64.b64encode(img.getvalue()).decode('utf-8'))
    return myhtml

@app.route('/decisionplotindividual',methods=['POST'])
def decisionplotindividual():
    global EXPLAINER,STUDENTS,Y_SEL,LABELS_Y
    st_obj = request.get_json()
    st_obj["id_student"] = int(st_obj["id_student"])
    k = STUDENTS.index(st_obj["id_student"])

    img = io.BytesIO()
    fig = shap.decision_plot(EXPLAINER["expected_value"],
     EXPLAINER["shap_values"][k],
     EXPLAINER["X"].loc[[st_obj["id_student"]]],
                   link='logit',show=False)       
    plt.savefig(img,bbox_inches='tight')
    plt.close()
    img.seek(0)
    my_html = '<img src="data:image/png;base64, {}">'.format(base64.b64encode(img.getvalue()).decode('utf-8'))
    return my_html

@app.route('/forceplot',methods=['GET'])
def forceplot():
    global EXPLAINER,Y_SEL,LABELS_Y
    return shap.force_plot(EXPLAINER["expected_value"], EXPLAINER["shap_values"], EXPLAINER["X"],link="logit").data


@app.route('/forceplotindividual',methods=['POST'])
def forceplotindividual():
    global EXPLAINER,STUDENTS,Y_SEL,LABELS_Y
    st_obj = request.get_json()
    st_obj["id_student"] = int(st_obj["id_student"])
    k = STUDENTS.index(st_obj["id_student"])
    #pickle.dump(EXPLAINER,open("EXPLAINER.p","wb"))
    #pickle.dump(STUDENTS,open("STUDENTS.p","wb"))
    #pickle.dump(LABELS_Y,open("LABELS_Y.p","wb"))
    #pickle.dump(k,open("id_student.p","wb"))
    #fig = 
    shap.force_plot(EXPLAINER["expected_value"], EXPLAINER["shap_values"][k], EXPLAINER["X"].iloc[k,:], link="logit",show=False,matplotlib=True).savefig('individual.png',format = "png",dpi = 150,bbox_inches = 'tight')
    #plt.savefig("individual.png")
    #plt.savefig("individual.png")
    return shap.force_plot(EXPLAINER["expected_value"], EXPLAINER["shap_values"][k], EXPLAINER["X"].iloc[k,:], link="logit").data       

@app.route('/dependenceplot',methods=['POST'])
def dependenceplot():
    dep_obj = request.get_json()
    #input(dep_obj)
    global EXPLAINER,Y_SEL,LABELS_Y
    img = io.BytesIO()
    fig = shap.dependence_plot(dep_obj['f1'], EXPLAINER["shap_values"], EXPLAINER["X"],interaction_index=dep_obj['f2'],show=False)
    plt.savefig(img)
    plt.close()
    img.seek(0)
    my_html = '<img src="data:image/png;base64, {}">'.format(base64.b64encode(img.getvalue()).decode('utf-8'))
    return my_html


if __name__ == "__main__":
    app.run(debug=True)
