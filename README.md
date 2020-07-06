# mitre-attack-mapper
Scripts for training and testing classifiers that map Splunk logs to MITRE ATT&amp;CK tactics and techniques.

## Dependencies

Run `sudo pip3 install -r requirements.txt` to install the required Python libraries.

## Usage

``` bash
    usage: main.py [-h] [--append_states {True,False}]
                   [--append_hosts {True,False}] [--model_type {nb,lsvc}]
                   [--target {tactics,techniques}] [--trial_prefix TRIAL_PREFIX]
                   [--ignore_singles {True,False}]
                   [--classify_dataset CLASSIFY_DATASET]
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
                            Bayes (nb) or Linear Support Vector Classifier (lsvc).
      --target {tactics,techniques}
                            The target class for testing, training, and
                            classification.
      --trial_prefix TRIAL_PREFIX
                            String to prefix to model files saved to disk.
      --ignore_singles {True,False}
                            Whether or not to remove tactics/techniques that
   			    only occur once from the dataset.
      --classify_dataset CLASSIFY_DATASET
                            Relative path to dataset (CSV) to use for
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

Trains a new model on all available data (no train/test split) with the same feature combination as the model specified by `--trial_prefix`. The new model and its parameters are saved to disk.

Command: `./main.py train <dataset> --model_type={nb, lsvc} --target={tactics, techniques} --append_states={True, False} --append_hosts={True, False} --trial_prefix=*`

#### Classify

Runs the `train` command to train a new model on `<dataset>` and then classifies the tactics/techniques for new data in `<classify_dataset>`. Predictions are saved to disk.

Command: `./main.py classify <dataset> --model_type={nb, lsvc} --target={tactics, techniques} --append_states={True, False} --append_hosts={True, False} --trial_prefix=* --classify_dataset=<classify_dataset>`

## Runtime

Runtime depends significantly on the number of CPU's available. For a system with 4 CPU's, `nb` runs in about 10 minutes and `lsvc` takes many, many hours. Running on the RC cluster with 16 CPU's drastically reduces runtime to a few minutes (`nb`) and a few hours (`lsvc`).

## Results

Results are saved to `./results/trial<#>/`. Each trial runs the `test` command for `nb, tactics`, `nb, techniques`, `lsvc, tactics`, and `lsvc, techniques`.

| Trial | IP Sub. | Timestamp Sub. | Append State | Append Host | NB Tact. Acc. | NB Tech. Acc. | LSVC Tact. Acc. | LSVC Tech. Acc. |
|-------|---------|----------------|--------------|-------------|---------------|---------------|-----------------|-----------------|
| 01    | False   | False          | False        | False       | 69.02%        | 66.11%        | 69.71%          | 68.43%          |
| 02    | False   | False          | True         | True        | 76.44%        | 73.73%        | 79.80%          | 77.65%          |
| 03    | False   | False          | True         | False       | 74.18%        | 73.17%        | 82.84%          | 76.54%          |
| 04    | False   | False          | False        | True        | 79.93%        | 71.80%        | 79.28%          | 77.71%          |
| 05    | False   | False          | False        | False       | 70.23%        | 73.57%        | 67.25%          | 75.92%          |
| 06    | False   | False          | True         | True        | 77.22%        | 80.85%        | 81.34%          | 84.45%          |
| 07    | False   | False          | True         | False       | 82.06%        | 70.51%        | 85.39%          | 85.11%          |
| 08    | False   | False          | False        | True        | 74.87%        | 77.24%        | 80.72%          | 82.50%          |

Full details of each trial are located in this [Google Sheet](https://docs.google.com/spreadsheets/d/1wbaUEhL4T0IqbtWG6HGlXXJpis3WfVbRUYDiIlfPDg4/edit#gid=137534009).
