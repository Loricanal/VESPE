# VISPE
The proposed architecture, namely VCS-based Explainable Student performance PrEdictor (VESPE), relies on a Flask (Python-based) application consisting of two main user interfaces, i.e., the input U.I. and output U.I., and of a core engine. The input UI allows end-users to specify (i) the input VCS usage data, (ii) the prediction targets, and (iii) the specification of the algorithm setup. The core engine consists of a set of supervised ML algorithms. The output U.I. provides end-users with visual explanations of the ML outcomes.

## Requirements
In order to use VESPE, **Python** and the **Pip** installer package must be installed on your machine.
Then you need to import all requirements and libraries running the following command:
```
pip install -r requirements.txt
```
## Input files


#### Data

The data file is written in CSV format. The columns represent the features names. Data values must be numeric.

A sample file used for classification is reported below.
|Feature 1|Feature 2|Feature 3|Feature 4|
|---------|---------|---------|---------|
|5.1      |3.5      |1.4      |0.2      |
|4.9      |3.0      |1.4      |0.2      |
|4.7      |3.2      |1.3      |0.2      |
|4.6      |3.1      |1.5      |0.2      |

A sample file used for regression is reported below.
|0  |1                 |2                  |
|---|------------------|-------------------|
|0.0380759064334241|0.0506801187398187|0.0616962065186885 |
|-0.00188201652779104|-0.044641636506989|-0.0514740612388061|
|0.0852989062966783|0.0506801187398187|0.0444512133365941 |
|-0.0890629393522603|-0.044641636506989|-0.0115950145052127|


#### Target

The target file is written in CSV format. The columns represent the target names. Data values must be numeric.

A sample file used for classification is reported below.
|Target 1|Target 2|
|--------|--------|
|0       |1       |
|0       |1       |
|0       |1       |
|1       |0       |
|1       |0       |
|1       |0       |
|2       |0       |
|2       |0       |
|2       |0       |

A sample file used for regression is reported below.
|Target|
|------|
|151.0 |
|75.0  |
|141.0 |
|206.0 |
|135.0 |
|97.0  |
|138.0 |

#### Specifications
The specifications file is written in JSON format. 

```json
{
	"features":{
		"families":{
		    ...
		}
	},
	"target":{
		"type":{
		    ...
		},
		"labels":{
		    ...
		}
	}
}
```

The core engine computes a model using all features by default; in addition  a model for each set of features specified in `families` is computed.
`type` allows you to define whether the targets should be used for classification or regression. In the first case, you can specify the label for each of the numerical targets values using `labels`.

A sample file used for classification is reported below.
```json
{
	"features":{
		"families":{
			"fam1":["Feature 1","Feature 2"],
			"fam2":["Feature 3","Feature 4"]
		}
	},
	"target":{
		"type":"classification",
		"labels":{
			"Target 1":["setosa","versicolor","virginica"],
			"Target 2":["setosa","other"]
		}
	}
}
```

A sample file used for regression is reported below.

```json
{
	"features":{
		"families":{
			"t1":["1","2","3","4","5"],
			"t2":["6","7","8","9"]
		}
	},
	"target":{
		"type":"regression"
	}
}
```

## Launch core engine
#### Manual start-up
Before running algorithms you have to pferform the following steps: open **datasets** folder, reate a new sub-folder related to the new dataset (e.g. *dataset 1*) and moving the input files inside.
The input files must be respectively called *data.csv* and *target.csv*. 
```
VESPE
│   README.md
│   compute_Models_Scores.py
|   compute_ShapModels.py
│   ...
└───datasets
│   │
│   └───dataset 
│       │   data.csv
│       │   target.csv
│       │   ...
...
```

The VESPE engine automatically splits train and test automatically. If you want to do it manually, you have to create two files for input data (*data_train.csv* and *data_test.csv*) rather than *data.csv* and two files for target (*target_train.csv* and *target_test.csv*).
```
VESPE
│   README.md
│   compute_Models_Scores.py
|   compute_ShapModels.py
│   ...
└───datasets
│   │
│   └───DATASETNAME
│       │   data.csv
│       │   target.csv
│       │   ...
...
```

###### Training ML models
The bash command below allows you to launch the calculation of machine learning models.
```
python3 compute_Models_Scores.py DATASETNAME
```

###### Training SHAP models
The bash command below allows you to launch the calculation of SHAP models.
```
python3 compute_ShapModels.py <DATASET NAME>
```
#### Automatic start-up with input UI
Let's enter in the folder **flask_app** then use the bash command below to launch the input UI.
```
python3 upload.py
```
The input UI allows you to upload data, targets and specification file. 

## Explainability

###### Launch output UI
Let's enter in the folder **flask_app** then use the bash command below to launch the output UI.
```
python3 app.py <DATASET NAME>
```

