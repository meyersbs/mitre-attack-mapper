# mitre-attack-mapper
Scripts for training and testing classifiers that map Splunk logs to MITRE ATT&amp;CK tactics and techniques.

## Dependencies

Run `sudo pip3 install -r requirements.txt` to install the required Python libraries.

## Usage

``` bash
    usage: main.py [-h] [--append_states {True,False}]
                   [--append_hosts {True,False}] [--model_type {nb,lsvc}]
                   [--target {tactics,techniques}]
                   {info,test,train,classify} dataset

    Map Splunk logs to MITRE ATT&CK states with Scikit-Learn.

    positional arguments:
      {info,test,train,classify}
                            The command to run. Info prints stats for the
                            specified dataset. Test does 10-fold CV to find the
                            best feature combinations. Train builds a new model
                            using all data and the best feature set. Classify
                            uses the best model to classify a new dataset.
      dataset               Relative path to the dataset (CSV) to use.

    optional arguments:
      -h, --help            show this help message and exit
      --append_states {True,False}
                            Whether or not to append tactics/techniques to the
                            raw data before testing/training. Default: False.
      --append_hosts {True,False}
                            Whether or not to append host types to the raw data
                            before testing/training. Default: False.
      --model_type {nb,lsvc}
                            The type of model to use. Either Multinomial Naive
                            Bayes (nb) or Linear Support Vector Classifier
                            (lsvc).
      --target {tactics,techniques}
                            The target class for testing, training, and
                            classification.
```

### Commands

#### Info

Prints out useful information about the dataset.

Command: `./main.py info <dataset>`

Sample Output:
```
Read 261 data points from '/home/benjamin/Code/mitre-attack-mapper/data/CPTC2018.csv'.
DATASET INFO:
  Data Points: 261
  TACTICS:
    'Discovery': 116
    ...
    'Credential Access, Lateral Movement, Lateral Movement': 23
    'Pre-Attack': 25
    ...
  TECHNIQUES:
    'Network Service Scanning': 105
    ...
    'Pre-Attack': 27
    'Brute Force, Valid Accounts, Remote Services': 26
    ...
```

#### Test

Runs 10-fold Cross Validation for the given `--model_type` using various combinations of features (see functions `testMNB(...)` and `testLSVC(...)`). Presents results for the best feature combination. Saves best model/parameters to disk. Current model types: `nb` ([multinomial naive bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive\_bayes.MultinomialNB.html)), `lsvc` ([linear support vector classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)).

Command: `./main.py test <dataset> --model_type={nb, lsvc} --target={tactics, techniques} --append_states={True, False} --append_hosts={True, False}`

Sample Output:

```
Best Score for MultinomialNB:
    0.6558823529411766
Best Parameters for MultinomialNB:
    clf__alpha: 0.01
    clf__fit_prior: True
    tfidf__use_idf: True
    vect__ngram_range: (1, 2)
Best Model Accuracy: 0.632183908045977
                                                       precision    recall  f1-score   support

         Brute Force, Valid Accounts, Remote Services       0.00      0.00      0.00         1
                                           Collection       0.00      0.00      0.00         1
                  Credential Access, Lateral Movement       0.44      0.44      0.44         9
Credential Access, Lateral Movement, Lateral Movement       0.78      0.70      0.74        10
                                            Discovery       0.85      0.83      0.84        35
                         Discovery, Credential Access       0.00      0.00      0.00         1
                                 Discovery, Discovery       0.00      0.00      0.00         4
                                            Execution       0.00      0.00      0.00         1
                          Execution, Lateral Movement       0.25      1.00      0.40         1
        Execution, Lateral Movement, Lateral Movement       0.00      0.00      0.00         1
                                       Initial Access       1.00      0.50      0.67         2
                                     Lateral Movement       0.29      0.40      0.33         5
                   Lateral Movement, Lateral Movement       0.40      1.00      0.57         2
                        Lateral Movement, Persistence       0.00      0.00      0.00         1
                Lateral Movement, Privlege Escalation       0.00      0.00      0.00         1
                                          Persistence       0.00      0.00      0.00         1
                                           Pre-Attack       0.60      0.90      0.72        10
                                 Privilege Escalation       0.00      0.00      0.00         1

                                             accuracy                           0.63        87
                                            macro avg       0.26      0.32      0.26        87
                                         weighted avg       0.60      0.63      0.60        87
```

#### Train

<b>NOTE:</b> Not yet implemented.

Trains a new model using the best feature combination and all of the available data in dataset.

#### Classify

<b>NOTE:</b> Not yet implemented.

Loads the best model from disk and classifies tactics/techniques for a new dataset.

## Runtime

Runtime depends significantly on the number of CPU's available. For a system with 4 CPU's, `nb` runs in about 10 minutes and `lsvc` takes many, many hours. Running on the RC cluster with 16 CPU's drastically reduces runtime to a few minutes (`nb`) and a few hours (`lsvc`).

## Results

Results are saved to `./results/trial<#>/`. Each trial runs the `test` command for `nb, tactics`, `nb, techniques`, `lsvc, tactics`, and `lsvc, techniques`.

| Trial | IP Sub. | Timestamp Sub. | Append State | Append Host | NB Tact. Acc. | NB Tech. Acc. | LSVC Tact. Acc. | LSVC Tech. Acc. |
|-------|---------|----------------|--------------|-------------|---------------|---------------|-----------------|-----------------|
| 01    | False   | False          | False        | False       | 67.2876%      | 68.3660%      | 66.2092%        | 71.1438%        |
| 02    | False   | False          | True         | True        | 71.8627%      | 70.7843%      | 81.0458%        | 78.7582%        |
| 03    | False   | False          | True         | False       | 78.2353%      | 71.2418%      | 79.9020%        | 81.1438%        |
| 04    | False   | False          | False        | True        | 67.2876%      | 69.5425%      | 70.0000%        | 72.4183%        |

Full details of each trial are located in this [Google Sheet](https://docs.google.com/spreadsheets/d/1wbaUEhL4T0IqbtWG6HGlXXJpis3WfVbRUYDiIlfPDg4/edit#gid=137534009).
