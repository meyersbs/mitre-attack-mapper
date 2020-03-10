# mitre-attack-mapper
Scripts for training and testing classifiers that map Splunk logs to MITRE ATT&amp;CK tactics and techniques.

## Dependencies

Run `sudo pip3 install -r requirements.txt` to install the required Python libraries.

## Usage

### Info

Running `./main.py info` will read in the data from `/data/CPTC2018.csv` and print out some useful information about the dataset.

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

### Train

Running `./main.py train <model_type> <target_label>` will train a simple bag-of-words model on the event data, and then run cross validation on many combinations of model parameters. The best model parameters and the classification report (precision, recall, F1-score, etc.) will be printed out.

Currently implemented model types are: `nb` ([naive bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive\_bayes.MultinomialNB.html)), `lsvc` ([linear SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)).

Valid target labels are: `tactics`, `techniques`.

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

## Runtimes

| Command  | Model         | Runtime     |
|----------|---------------|-------------|
| `nb`     | MultinomialNB | ~2 minutes  |
| `lsvc`   | LinearSVC     | ~25 minutes |
| `linear` | Various       | too long    |

## Results

### MultinomialNB

### LinearSVC
