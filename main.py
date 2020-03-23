#!/usr/bin/env python3

#### PYTHON IMPORTS ############################################################
import csv
import json
import numpy as NP
import os
import sys
import warnings

from collections import Counter


#### SCIKITLEARN IMPORTS #######################################################
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
warnings.filterwarnings("ignore", category=UserWarning)


#### GLOBALS ###################################################################
CURR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CURR_PATH, "data")
DATA_FILE = os.path.join(DATA_PATH, "CPTC2018.csv")
MODELS_PATH = os.path.join(CURR_PATH, "models")


#### HELPER FUNCTIONS ##########################################################
def readData(filename):
    """
    Read data from a CSV file. The CSV file should have the following 4 columns:
      1 ID          Unique ID for event
      2 HOST        The hostname that the Splunk event ocurred on
      3 HOST_TYPE   One of "Attacker" or "Victim"
      4 EVENT       Raw text of a Splunk event (log)
      5 TACTIC      One or more MITRE ATT&CK Tactics
      6 TECHNIQUE   One or more MITRE ATT&CK Techniques

    GIVEN:
      filename (str)    file path to read data from

    RETURN:
      data (list)       list of events from CSV file
      hosts (list)      list of host types for each event
      labels (dict)     dictionary with two keys, "tactics" and "techniques";
                        each value is a list of human-annotated labels
    """
    data = list()
    hosts = list()
    labels = {"tactics": list(), "techniques": list()}
    with open(filename, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",", quotechar="\"")

        # Skip header row
        next(csv_reader)

        # Read data from CSV
        for row in csv_reader:
            data.append(row[3])
            hosts.append(row[2])
            labels["tactics"].append(row[4])
            labels["techniques"].append(row[5])

    print("Read {} data points from '{}'.".format(len(data), filename))
    return data, hosts, labels


def printStats(data, labels):
    """ Print useful information about the data. """
    cnt_tactics = Counter(labels["tactics"])
    k_tactics = list(cnt_tactics.keys())
    v_tactics = list(cnt_tactics.values())
    cnt_techniques = Counter(labels["techniques"])
    k_techniques = list(cnt_techniques.keys())
    v_techniques = list(cnt_techniques.values())

    print("DATASET INFO:")
    print("  Data Points: {}".format(len(data)))
    print("  TACTICS:")
    for i in range(0, len(k_tactics)):
        print("    '{}': {}".format(k_tactics[i], v_tactics[i]))
    print("  TECHNIQUES:")
    for i in range(0, len(k_techniques)):
        print("    '{}': {}".format(k_techniques[i], v_techniques[i]))


def preprocessData(data):
    """
    Clean up the data by replacing timestamps and IP addresses with unique
    tokens.

    GIVEN:
      data (list)       list of events

    RETURN:
      new_data (list)   copy of `data` with timestamps and IP addresses
                        cleaned up
    """
    new_data = data

    # Replace timestamps with '<<TIMESTAMP>>'
    # Replace IP addresses with '<<ATTACKER>>' or '<<DEFENDER>>'

    return new_data


def augmentData(data, hosts, labels, target_class):
    """
    Add the Tactic or Technique to the data. Add host type to the data.
    
    GIVEN:
      data (list)           list of events
      hosts (list)          one of ["Attacker", "Victim"]
      labels (dict)         human-annotated labels for data
      target_class (str)    one of ["tactics", "techniques"]

    RETURN:
      new_data (list)       copy of `data` with tactics/techniques appended
    """
    new_data = data

    if target_class == "tactics":
        for i in range(0, len(data)):
            new_data[i] = new_data[i] + " " + labels["techniques"][i] + " " + hosts[i]
    else:
        for i in range(0, len(data)):
            new_data[i] = new_data[i] + " " + labels["tactics"][i] + " " + hosts[i]

    return new_data


def trainTestSplit(data, labels, target_class, test_size):
    """
    Split the data and it's labels into training and testing sets.

    GIVEN:
      data (list)           list of events
      labels (dict)         human-annotated labels for data
      target_class (str)    one of ["tactics", "techniques"]
      test_size (double)    percentage to hold out for testing

    RETURN:
      X_train (list)    list of training data values
      X_test (list)     list of testing data values
      Y_train (list)    list of training data labels
      Y_test (list)     list of testing data labels
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        data, labels[target_class], test_size=test_size,
        #stratify=labels[target_class]
    )

    return X_train, X_test, Y_train, Y_test


def trainMNB(data, labels, target_class, test_size):
    """
    Train and test the performance of MultinomialNB with various parameter
    combinations.
    """
    X_train, X_test, Y_train, Y_test = trainTestSplit(
        data, labels, target_class, test_size)

    pipe_MNB = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB())
    ])
    
    # All combinations of these parameters will be tested
    GS_MNB_params = {
        "vect__ngram_range": [(1, 1), (1, 2), (1, 3)], # Unigrams, Bigrams, Trigrams
        "vect__stop_words": (None, "english"),
        "vect__strip_accents": (None, "ascii", "unicode"),
        "vect__analyzer": ("word", "char", "char_wb"),
        "vect__lowercase": (True, False),
        "tfidf__use_idf": (True, False),
        "clf__alpha": (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99),
        "clf__fit_prior": (True, False)
    }

    # Test all combinations of GS_MNB_params using 10-fold CV and all available CPU's
    pipe_GS_MNB = GridSearchCV(pipe_MNB, GS_MNB_params, cv=10, n_jobs=-1, verbose=1)
    pipe_GS_MNB.fit(X_train, Y_train)

    best_params = pipe_GS_MNB.best_params_
    print("Best Score for MultinomialNB:")
    print("    {}".format(pipe_GS_MNB.best_score_))
    print("Best Parameters for MultinomialNB:")
    for p in sorted(GS_MNB_params.keys()):
        print("    {}: {}".format(p, best_params[p]))

    #print(pipe_GS_MNB.cv_results_)

    pipe_GS_MNB_predicted = pipe_GS_MNB.predict(X_test)
    print(classification_report(Y_test, pipe_GS_MNB_predicted))

    # Dump best model to disk
    outfile = os.path.join(MODELS_PATH, "MultinomialNB_{}_best.pkl".format(target_class))
    joblib.dump(pipe_GS_MNB.best_estimator_, outfile)
    # Dump best model parameters to disk
    params_outfile = os.path.join(MODELS_PATH, "MultinomialNB_{}_best_params.json".format(target_class))
    f = open(params_outfile, "w")
    f.write(json.dumps(best_params))
    f.close()


def trainLSVC(data, labels, target_class, test_size):
    """
    Train and test the performance of LinearSVC with various parameter
    combinations.
    """
    X_train, X_test, Y_train, Y_test = trainTestSplit(
        data, labels, target_class, test_size)

    pipe_LSVC = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", LinearSVC())
    ])

    # All combinations of these parameters will be tested
    GS_LSVC_params = {
        "vect__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "vect__stop_words": (None, "english"),
        "vect__strip_accents": (None, "ascii", "unicode"),
        "vect__analyzer": ("word", "char", "char_wb"),
        "vect__lowercase": (True, False),
        "tfidf__use_idf": (True, False),
        "clf__penalty": ("l1", "l2"),
        "clf__loss": ("hinge", "squared_hinge"),
        "clf__dual": (True, False),
        "clf__tol": (1e-1, 1e-4, 1e-10),
        "clf__multi_class": ("ovr", "crammer_singer"),
        "clf__fit_intercept": (True, False),
        "clf__max_iter": (1000, 5000)
    }

    # Test all combinations of GS_LSV_params using 10-fold CV and all available CPU's
    pipe_GS_LSVC = GridSearchCV(pipe_LSVC, GS_LSVC_params, cv=10, n_jobs=-1, verbose=1)
    pipe_GS_LSVC.fit(X_train, Y_train)

    best_params = pipe_GS_LSVC.best_params_
    print("Best Score for LinearSVC:")
    print("    {}".format(pipe_GS_LSVC.best_score_))
    print("Best Parameters for LinearSVC:")
    for p in sorted(GS_LSVC_params.keys()):
        print("    {}: {}".format(p, best_params[p]))

    pipe_GS_LSVC_predicted = pipe_GS_LSVC.predict(X_test)
    print(classification_report(Y_test, pipe_GS_LSVC_predicted))

    # Dump best model to disk
    outfile = os.path.join(MODELS_PATH, "LinearSVC_{}_best.pkl".format(target_class))
    joblib.dump(pipe_GS_LSVC.best_estimator_, outfile)
    # Dump best model parameters to disk
    params_outfile = os.path.join(MODELS_PATH, "LinearSVC_{}_best_params.json".format(target_class))
    f = open(params_outfile, "w")
    f.write(json.dumps(best_params))
    f.close()


#### MAIN ######################################################################
if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "info":
        data, hosts, labels = readData(DATA_FILE)
        printStats(data, labels)
    elif args[0] == "train":
        data, hosts, labels = readData(DATA_FILE)
        if args[1] == "nb":
            if args[3] == "yes":
                data = augmentData(data, hosts, labels, args[2])
            trainMNB(data, labels, args[2], 0.33)
        elif args[1] == "lsvc":
            if args[3] == "yes":
                data = augmentData(data, hosts, labels, args[2])
            trainLSVC(data, labels, args[2], 0.33)

