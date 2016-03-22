#!/usr/bin/env python

import numpy as np
import matplotlib as plt
import csv
import argparse
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB



def get_raw_data(fname):
    titles = None
    data = []
    with open(fname) as f:
        c = csv.reader(f)
        titles = next(c)
        for line in c:
            data.append(line)
    return titles, data

def get_fields(titles, data, fields):
    titles = [titles[x] for x in fields]
    data = [[line[x] for x in fields] for line in data]
    data = np.array(data)
    return titles, data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input csv data file")
    parser.add_argument("-x", "--features", help="indices of feature fields", nargs="+", type=int)
    parser.add_argument("-y", "--target", help="index of ground truth field", type=int)
    parser.add_argument("--benchmark", help="benchmark target field", type=int)
    parser.add_argument("--score",help="scoring metric for determining best model. ", default="f1")
    parser.add_argument("-o", "--output", help="output model file, pickled")
    parser.add_argument("-i","--image", help="apply model to classify an image")

    return parser.parse_args()



if __name__ == "__main__":

    rand_state = 42

    args = get_args()
    raw_titles, raw_data = get_raw_data(args.input)
    feature_title, X = get_fields(raw_titles, raw_data, args.features)
    target_title, y = get_fields(raw_titles, raw_data, [args.target])
    X = X.astype(float)
    y = y.astype(float)

    y_prime = np.zeros_like(y)
    if args.benchmark:
        benchmark_title, y_prime = get_fields(\
                raw_titles, raw_data, [args.benchmark])
        le = LabelEncoder()
        le.fit(["False", "True"])
        y_prime = le.transform(y_prime[:,0])
    else:
        y_prime = y_prime[:,0] # added by PS to make shapes match

    y = y[:,0]

    
    X = X[y>-1,:]
    y_prime = y_prime[y>-1]
    y = y[y>-1]

    #test split both target and benchmark
    y_tmp = np.vstack([y, y_prime]).T
    X_train, X_test, y_train, y_test = train_test_split(\
            X, y_tmp, test_size = 1/4.0, random_state = rand_state)
    y_train, y_prime_train = (y_train[:,0], y_train[:,1])
    y_test, y_prime_test = (y_test[:,0], y_test[:,1])

    
    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)


    classifiers = [
            KNeighborsClassifier(),
            SVC(),
            RandomForestClassifier(),
            GaussianNB()]

    names = ["KNN", "GaussianSVM", "RandomForest", "Gaussian Naive Bayes",
             "VotingClassifier of KNN, GaussianSVM, and RF"]

    params = [{"n_neighbors": 3. ** np.arange(5)},
              {"kernel": ["rbf"], "gamma": 3. ** np.arange(-5, 5), 
               "C": 3. ** np.arange(-5,5)},
              {"n_estimators": 3 ** np.arange(7), 
               "max_features": np.arange(1, X.shape[1])},
              {},
              ]

    best_estimators = []
    best_scores = []
    best_params = []
    
    for name, param, classifier in zip(names, params, classifiers):
        clf = GridSearchCV(classifier, param, cv=3, scoring=args.score, verbose=3, n_jobs=8)
        clf.fit(X_train_norm, y_train)
        best_estimators.append(clf.best_estimator_)
        best_scores.append(clf.best_score_)
        best_params.append(clf.best_params_)

    #vote from the best
    voting_clf = VotingClassifier(list(zip(names, best_estimators))[:3], voting="hard")
    score = cross_val_score(voting_clf, X_train_norm, y_train, cv=3, scoring=args.score)

    best_estimators.append(voting_clf)
    best_scores.append(score.mean())
    best_params.append({})
    
    best_model_idx = np.argmax(np.array(best_scores))

    best_model = best_estimators[best_model_idx]

    print("Best model is: {}, with parameters {}".format(\
            names[best_model_idx], 
            repr(best_params[best_model_idx])))
    print("Validation F1 Score: {}".format(best_scores[best_model_idx]))

    #retrain with all of the training sample
    best_model.fit(X_train_norm, y_train)
    pred = best_model.predict(X_test_norm)
    print("Testing F1 Score: {}".format(f1_score(y_test, pred)))

    if args.benchmark:
        print("Benchmark F1 Score: {}".format(\
                f1_score(y_test, y_prime_test)))

    if args.output:
        with open(args.output, "wb") as fo:
            pickle.dump(best_model, fo)
            print("Pickled the best model in {}".format(args.output))
    
    print("target: {}").format(target_title)
    print("features tried {}").format(feature_title)

    
