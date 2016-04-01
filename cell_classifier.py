#!/usr/bin/env python

import numpy as np
import matplotlib as plt
import csv
import argparse
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, precision_recall_fscore_support

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import pandas as pd
import skimage
import scipy.misc, scipy.interpolate, scipy.ndimage
from skimage import exposure, filters, morphology, feature, segmentation


model_names = ["knn", "gsvm", "rf", "gnb", "lda", "voting"]

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
    parser.add_argument("--score",help="scoring metric for determining best model", default="f1")
    parser.add_argument("-o", "--output", help="output model file, pickled")
    parser.add_argument("-i","--image", help="apply model to classify an image")
    parser.add_argument("--select-model", help="The model classes to train, defaults to all.", nargs="*", choices=model_names)
    parser.add_argument("--unknown-as", help="Treat unknown label as", choices=["0","1","-1","remove"], default="remove")

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

    if args.unknown_as == "remove":
        X = X[y>-1,:]
        y_prime = y_prime[y>-1]
        y = y[y>-1]
    elif args.unknown_as in ["0", "1", "-1"]:
        y[y==-1] = int(args.unknown_as)


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
            GaussianNB(),
            LDA(),
            ]

    names = model_names 
           

    params = [{"n_neighbors": 3. ** np.arange(5)},
              {"kernel": ["rbf"], "gamma": 3. ** np.arange(-5, 5), 
               "C": 3. ** np.arange(-5,5)},
              {"n_estimators": 3 ** np.arange(6), 
               "max_features": np.arange(1, X.shape[1])},
              {},
              {},
              ]

    best_estimators = []
    best_scores = []
    best_params = []
    selected_names = []
    
    for name, param, classifier in zip(names, params, classifiers):
        if args.select_model and name in args.select_model and name != "voting":
            selected_names.append(name)
            #lda tend to hang if run parallel
            clf = GridSearchCV(classifier, param, cv=3, scoring=args.score, verbose=3, n_jobs=1)
            clf.fit(X_train_norm, y_train)
            best_estimators.append(clf.best_estimator_)
            best_scores.append(clf.best_score_)
            best_params.append(clf.best_params_)
            

    if "voting" in args.select_model:
        selected_names.append("voting")
        #vote from the best
        voting_clf = VotingClassifier(list(zip(names, best_estimators)), voting="hard")
        score = cross_val_score(voting_clf, X_train_norm, y_train, cv=3, scoring=args.score)

        best_estimators.append(voting_clf)
        best_scores.append(score.mean())
        best_params.append({})

    print(zip(selected_names,best_scores))
    print("LDA coef:", zip(feature_title, LDA().fit(X_train_norm, y_train).coef_[0]))
    
    best_model_idx = np.argmax(np.array(best_scores))

    best_model = Pipeline([("normalize", normalizer), ("estimator", best_estimators[best_model_idx])])

    print("Best model is: {}, with parameters {}".format(\
            selected_names[best_model_idx], 
            repr(best_params[best_model_idx])))
    print("Validation F1 Score: {}".format(best_scores[best_model_idx]))

    #retrain with all of the training sample
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    print("Testing precision: {}, recall: {}, F1 Score: {}, support: {}".format(*precision_recall_fscore_support(y_test, pred, average="binary")))

    if args.benchmark:
        print("Benchmark precision: {}, recall: {}, F1 score: {}, support: {}".format(*precision_recall_fscore_support(y_test, y_prime_test, average="binary")))

    if args.output:
        with open(args.output, "wb") as fo:
            pickle.dump(best_model, fo)
            print("Pickled the best model in {}".format(args.output))
    
    print("target: {}").format(target_title)
    print("features tried {}").format(feature_title)

    if args.image:
        disksize = 5; corrthres = 98;
        print "using settings which may no longer be used. "
        print "e.g. disk size {}, correlation threshold {} percentile".format(disksize, corrthres)
        imgin = skimage.img_as_uint(scipy.misc.imread(args.image))
        minp = np.percentile(imgin, 50)
        img_iadjust = imgin*(imgin >= minp)  
        tmplt = morphology.disk(disksize)
        tmplt_matched = skimage.feature.match_template(img_iadjust, tmplt, pad_input=True)
        tmplt_thresholdbinary = tmplt_matched >= np.percentile(tmplt_matched, corrthres)
        img_lbl, ncell = scipy.ndimage.measurements.label(tmplt_thresholdbinary,np.ones((3,3), bool))
        print "Detected {} objects".format(ncell)
        columns = ['otherindex','indexcolumnidislike','imgname','imgnumber','roi','id','label',
                'coords','bbox_rmin', 'bbox_rmax','bbox_cmin', 'bbox_cmax','centroidr','centroidc',
                'meani','equivdiameter','circularity','eccentricity','area','minor_axis_length',
                'major_axis_length','min_intensity','max_intensity']
        lbldetails = pd.DataFrame(columns=columns)
        lblprops = skimage.measure.regionprops(img_lbl,img_iadjust)
        z=0
        for ilbl in lblprops:
            circularity = (ilbl.perimeter*ilbl.perimeter) / (np.pi*4.0*ilbl.area)
            lbldetails.loc[z] = ["NA","NA","NA",
                                "NA", "NA", z, 
                                ilbl.label, 
                                ilbl.coords,
                                ilbl.bbox[0], ilbl.bbox[2], ilbl.bbox[1], ilbl.bbox[3],
                                ilbl.centroid[0], ilbl.centroid[1], 
                                ilbl.mean_intensity, 
                                ilbl.equivalent_diameter,
                                circularity, 
                                ilbl.eccentricity,
                                ilbl.area, 
                                ilbl.minor_axis_length, ilbl.major_axis_length,
                                ilbl.min_intensity, ilbl.max_intensity]
            z += 1
        
        raw_titles = lbldetails.columns.tolist()
        raw_data = lbldetails.values.tolist()
        print raw_titles
        print args.features
        feature_title, X = get_fields(raw_titles, raw_data, args.features )
        X = X.astype(float)


        lbldetails['classified'] = best_model.predict(X)
        # add an object to the front (represents label number 0)
        lbl_classified = pd.Series([0])
        lbl_classified = lbl_classified.append(lbldetails['classified'])
        # we want objects removed to be labeled 1 (aka True). They are 0 right now
        lbl_classified = 1 - lbl_classified
        lbl_classified = np.abs(lbl_classified)
        lbl_classified = np.array(lbl_classified.astype(bool))
        remove_pixel = lbl_classified[img_lbl]
        img_lbl[remove_pixel] = 0
        filename = os.path.splitext(os.path.basename(args.image))[0]
        print filename+"_cells.tif"
        scipy.misc.imsave(filename+"_cells.tif", img_lbl > 0.5)
