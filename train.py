import h5py
import numpy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

import xgboost

def load_array(file, name):
    array = file[name]
    array = numpy.asarray(array)
    return array

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "input hdf5 file", type=str, default="features.hdf5")
parser.add_argument("--rank", help = "taxonomic rank features for training", type=str, default="phylum")
parser.add_argument("--biopsy_loc", help = "biopsy location to take taxonomic data from", type=str, default="TI")
args = parser.parse_args()
 
f_in = h5py.File(args.input, "r")
y = load_array(f_in, "label_%s_%s" % (args.biopsy_loc, args.rank))
X = load_array(f_in, "features_%s_%s" % (args.biopsy_loc, args.rank))

param = {
    "max_depth" : 3,
    "eta" : 0.1,
    "min_child_weight" : 20,
    'eval_metric': 'logloss',
    'colsample_bylevel' : 1.0,
}

n_kfold = 50

results = {}
for method in ["RandomForest", "BDT"]:
    results[method] = {}
    for split in ["test", "train"]:
        results[method][split] = { "fpr" : [], "tpr" : [], "auc" : [] }

do_tax = False
if do_tax:
    for i in range(n_kfold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

        classifier = RandomForestClassifier(n_estimators=100, max_depth=4)

        #print("Training random forest")
        classifier.fit(X_train,y_train)
        prediction_train = classifier.predict_proba(X_train)
        prediction_test = classifier.predict_proba(X_test)

        fpr_test, tpr_test, thresh_test = roc_curve(y_test, prediction_test[:,1])
        fpr_train, tpr_train, thresh_train = roc_curve(y_train, prediction_train[:,1])

        fpr_mean = numpy.linspace(0, 1, 100)
        tpr_test_interp = numpy.interp(fpr_mean, fpr_test, tpr_test)
        tpr_train_interp = numpy.interp(fpr_mean, fpr_train, tpr_train)

        auc_test = auc(fpr_test, tpr_test)
        auc_train = auc(fpr_train, tpr_train)

        results["RandomForest"]["test"]["fpr"].append(fpr_mean)
        results["RandomForest"]["test"]["tpr"].append(tpr_test_interp)
        results["RandomForest"]["test"]["auc"].append(auc_test)
        results["RandomForest"]["train"]["fpr"].append(fpr_mean)
        results["RandomForest"]["train"]["tpr"].append(tpr_train_interp)
        results["RandomForest"]["train"]["auc"].append(auc_train) 

        bdt = xgboost.train(param, xgboost.DMatrix(X_train, label = y_train), 50)

        pred_train = bdt.predict(xgboost.DMatrix(X_train, label = y_train))
        pred_test = bdt.predict(xgboost.DMatrix(X_test, label = y_test))

        fpr_test, tpr_test, thresh_test = roc_curve(y_test, pred_test)
        fpr_train, tpr_train, thresh_train = roc_curve(y_train, pred_train)

        tpr_test_interp = numpy.interp(fpr_mean, fpr_test, tpr_test)
        tpr_train_interp = numpy.interp(fpr_mean, fpr_train, tpr_train)

        auc_test = auc(fpr_test, tpr_test)
        auc_train = auc(fpr_train, tpr_train)

        results["BDT"]["test"]["fpr"].append(fpr_mean)
        results["BDT"]["test"]["tpr"].append(tpr_test_interp)
        results["BDT"]["test"]["auc"].append(auc_test)
        results["BDT"]["train"]["fpr"].append(fpr_mean)
        results["BDT"]["train"]["tpr"].append(tpr_train_interp)
        results["BDT"]["train"]["auc"].append(auc_train) 


train_fracs = numpy.linspace(0.1, 0.8, 20) 
results_lc = {}
for method in ["RandomForest", "BDT"]:
    results_lc[method] = {}
    for split in ["test", "train"]:
        results_lc[method][split] = {}
        for frac in train_fracs:
            results_lc[method][split][frac] = { "fpr" : [], "tpr" : [], "auc" : [] }

do_learning_curve = True
if do_learning_curve:
    for train_frac in train_fracs: 
        for i in range(n_kfold):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_frac)
            classifier = RandomForestClassifier(n_estimators=100, max_depth=4)

            classifier.fit(X_train,y_train)
            prediction_train = classifier.predict_proba(X_train)
            prediction_test = classifier.predict_proba(X_test)

            fpr_test, tpr_test, thresh_test = roc_curve(y_test, prediction_test[:,1])
            fpr_train, tpr_train, thresh_train = roc_curve(y_train, prediction_train[:,1])

            fpr_mean = numpy.linspace(0, 1, 100)
            tpr_test_interp = numpy.interp(fpr_mean, fpr_test, tpr_test)
            tpr_train_interp = numpy.interp(fpr_mean, fpr_train, tpr_train)

            auc_test = auc(fpr_test, tpr_test)
            auc_train = auc(fpr_train, tpr_train)

            results_lc["RandomForest"]["test"][train_frac]["fpr"].append(fpr_mean)
            results_lc["RandomForest"]["test"][train_frac]["tpr"].append(tpr_test_interp)
            results_lc["RandomForest"]["test"][train_frac]["auc"].append(auc_test)
            results_lc["RandomForest"]["train"][train_frac]["fpr"].append(fpr_mean)
            results_lc["RandomForest"]["train"][train_frac]["tpr"].append(tpr_train_interp)
            results_lc["RandomForest"]["train"][train_frac]["auc"].append(auc_train)

            bdt = xgboost.train(param, xgboost.DMatrix(X_train, label = y_train), 50)

            pred_train = bdt.predict(xgboost.DMatrix(X_train, label = y_train))
            pred_test = bdt.predict(xgboost.DMatrix(X_test, label = y_test))

            fpr_test, tpr_test, thresh_test = roc_curve(y_test, pred_test)
            fpr_train, tpr_train, thresh_train = roc_curve(y_train, pred_train)

            tpr_test_interp = numpy.interp(fpr_mean, fpr_test, tpr_test)
            tpr_train_interp = numpy.interp(fpr_mean, fpr_train, tpr_train)

            auc_test = auc(fpr_test, tpr_test)
            auc_train = auc(fpr_train, tpr_train)

            results_lc["BDT"]["test"][train_frac]["fpr"].append(fpr_mean)
            results_lc["BDT"]["test"][train_frac]["tpr"].append(tpr_test_interp)
            results_lc["BDT"]["test"][train_frac]["auc"].append(auc_test)
            results_lc["BDT"]["train"][train_frac]["fpr"].append(fpr_mean)
            results_lc["BDT"]["train"][train_frac]["tpr"].append(tpr_train_interp)
            results_lc["BDT"]["train"][train_frac]["auc"].append(auc_train)



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.yaxis.set_ticks_position('both')
ax1.grid(True)

colors = { "RandomForest" : "blue", "BDT" : "red" }

if do_tax:
    for method in ["RandomForest", "BDT"]:
        tpr_mean = numpy.mean(results[method]["test"]["tpr"], axis=0)
        tpr_std = numpy.std(results[method]["test"]["tpr"], axis=0)
        fpr_mean = numpy.mean(results[method]["test"]["fpr"], axis=0)
        auc_mean = numpy.mean(results[method]["test"]["auc"])
        auc_std = numpy.std(results[method]["test"]["auc"])

        ax1.plot(fpr_mean, tpr_mean, color = colors[method], label = "%s AUC: %.2f +/- %.2f" % (method, auc_mean, auc_std))
        ax1.fill_between(fpr_mean, tpr_mean - tpr_std, tpr_mean + tpr_std, color = colors[method], alpha = 0.25, label = r'$\pm 1\sigma$')

    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc = "lower right")
    plt.savefig("roc_comparison_%s_%s.pdf" % (args.rank, args.biopsy_loc))

if do_learning_curve:
    for method in ["RandomForest", "BDT"]:
        x = train_fracs
        y = []
        y_err = []
        for val in train_fracs:
            auc_mean = numpy.mean(results_lc[method]["test"][val]["auc"])
            auc_std = numpy.std(results_lc[method]["test"][val]["auc"])
            y.append(auc_mean)
            y_err.append(auc_std)

        ax1.errorbar(train_fracs, y, yerr = y_err, label = method, color = colors[method], marker = 'o')

    plt.xlabel("Training fraction")
    plt.ylabel("AUC")
    plt.legend(loc = "lower right")
    plt.savefig("learning_curve_%s_%s.pdf" % (args.rank, args.biopsy_loc))
