import h5py
import numpy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

import xgboost

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

bdt_param = {
    "max_depth" : 3,
    "eta" : 0.1,
    "min_child_weight" : 20,
    'eval_metric': 'logloss',
    'colsample_bylevel' : 1.0,
}

rf_param = {
    "n_estimators" : 100,
    "max_depth" : 4,
}

colors = { "rf" : "blue", "bdt" : "red" }
plot_labels = { "rf" : "RandomForest", "bdt" : "BDT" }

def calc_auc(y, pred, interp = 100):
    fpr, tpr, thresh = roc_curve(y, pred)

    fpr_interp = numpy.linspace(0, 1, interp)
    tpr_interp = numpy.interp(fpr_interp, fpr, tpr)

    auc_ = auc(fpr, tpr)

    return fpr_interp, tpr_interp, auc_


class TreeTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.tag = kwargs.get('tag', '')

        self.X = kwargs.get('X', [])
        self.y = kwargs.get('y', [])
    
        self.train_frac = kwargs.get('train_frac', 0.7)
        
        self.bdt_param = kwargs.get('bdt_param', bdt_param)
        self.rf_param = kwargs.get('rf_param', rf_param)

        self.bdt_results = { "fpr" : { "test" : [], "train" : [] },
                             "tpr" : { "test" : [], "train" : [] },
                             "auc" : { "test" : [], "train" : [] }
                           }
        self.rf_results = { "fpr" : { "test" : [], "train" : [] },
                             "tpr" : { "test" : [], "train" : [] },
                             "auc" : { "test" : [], "train" : [] }
                           }

        self.n_kfold = kwargs.get('n_kfold', 50)


    def set_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = self.train_frac)

    def train_rf(self):
        classifier = RandomForestClassifier(n_estimators = self.rf_param['n_estimators'],
                                            max_depth = self.rf_param['max_depth'] )
        
        classifier.fit(self.X_train, self.y_train)
        rf_pred_train = classifier.predict_proba(self.X_train)[:,1]
        rf_pred_test = classifier.predict_proba(self.X_test)[:,1]
        
        fpr, tpr, auc_ = calc_auc(self.y_train, rf_pred_train)
        self.rf_results["fpr"]["train"].append(fpr)
        self.rf_results["tpr"]["train"].append(tpr)
        self.rf_results["auc"]["train"].append(auc_)
        
        fpr, tpr, auc_ = calc_auc(self.y_test, rf_pred_test)
        self.rf_results["fpr"]["test"].append(fpr)
        self.rf_results["tpr"]["test"].append(tpr)
        self.rf_results["auc"]["test"].append(auc_)

    def train_bdt(self):
        bdt = xgboost.train(self.bdt_param, xgboost.DMatrix(self.X_train, 
                                                            label = self.y_train))
        
        bdt_pred_train = bdt.predict(xgboost.DMatrix(self.X_train, label = self.y_train))
        bdt_pred_test = bdt.predict(xgboost.DMatrix(self.X_test, label = self.y_test))

        fpr, tpr, auc_ = calc_auc(self.y_train, bdt_pred_train)
        self.bdt_results["fpr"]["train"].append(fpr)
        self.bdt_results["tpr"]["train"].append(tpr)
        self.bdt_results["auc"]["train"].append(auc_)
        
        fpr, tpr, auc_ = calc_auc(self.y_test, bdt_pred_test)
        self.bdt_results["fpr"]["test"].append(fpr)
        self.bdt_results["tpr"]["test"].append(tpr)
        self.bdt_results["auc"]["test"].append(auc_)

    def train(self):
        self.results = { "bdt" : {}, "rf" : {} }

        for i in range(self.n_kfold):
            self.set_data()
            self.train_rf()
            self.train_bdt()
    
        for type in ["train", "test"]:
            self.results["bdt"][type] = { "fpr_mean" : numpy.mean(self.bdt_results["fpr"][type], axis=0),
                                          "fpr_std" : numpy.std(self.bdt_results["fpr"][type], axis=0),
                                          "tpr_mean" : numpy.mean(self.bdt_results["tpr"][type], axis=0),
                                          "tpr_std" : numpy.std(self.bdt_results["tpr"][type], axis=0), 
                                          "auc_mean" : numpy.mean(self.bdt_results["auc"][type]),
                                          "auc_std" : numpy.std(self.bdt_results["auc"][type]),
                                        }
            self.results["rf"][type] = { "fpr_mean" : numpy.mean(self.rf_results["fpr"][type], axis=0),
                                          "fpr_std" : numpy.std(self.rf_results["fpr"][type], axis=0),
                                          "tpr_mean" : numpy.mean(self.rf_results["tpr"][type], axis=0),
                                          "tpr_std" : numpy.std(self.rf_results["tpr"][type], axis=0), 
                                          "auc_mean" : numpy.mean(self.rf_results["auc"][type]),
                                          "auc_std" : numpy.std(self.rf_results["auc"][type]),
                                        } 

    def make_plots(self):
        for type in ["train", "test"]:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.yaxis.set_ticks_position('both')
            ax1.grid(True)
            
            for method in ["bdt", "rf"]:
                ax1.plot(self.results[method][type]["fpr_mean"], 
                         self.results[method][type]["tpr_mean"], 
                         color = colors[method], 
                         label = "%s AUC: %.2f +/- %.2f" % (plot_labels[method], self.results[method][type]["auc_mean"], self.results[method][type]["auc_std"]))
                ax1.fill_between(self.results[method][type]["fpr_mean"],
                                 self.results[method][type]["tpr_mean"] - (self.results[method][type]["tpr_std"]/2.),
                                 self.results[method][type]["tpr_mean"] + (self.results[method][type]["tpr_std"]/2.),
                                 color = colors[method],
                                 alpha = 0.25, label = r'$\pm 1\sigma$')
            
            plt.xlim([-0.05,1.05])
            plt.ylim([-0.05,1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc = "lower right")
            plt.savefig("plots/roc_comparison_%s_%s.pdf" % (self.tag, type))
            plt.clf()
