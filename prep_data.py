import os, sys
import h5py

import biom
import pandas
import numpy

import xgboost

from scipy import interp

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

ranks = {
    'phylum': 2,
    'class': 3,
    'order': 4,
    'family': 5,
    'species': 6,
}

def collapse_to(lst, item, default='Unassigned'):
    if len(lst)> item:
        return lst[item]
    else:
        return default

def load_b(files):
    b = []
    for file in files:
        b.append(biom.load_table(file))

    b_all = b[0]
    for i in range(1,len(b)):
        b_all = b_all.merge(b[i], sample='union', observation='union')

    b_norm = b_all.norm(axis = 'sample', inplace=False)

    b_rank = {}
    for r,n in ranks.items():
        b_rank[r] = b_norm.collapse(lambda id_, md: collapse_to(md['taxonomy'], n, 'Unassigned'), axis='observation', norm=False).to_dataframe(dense=True)
    b_rank['zotu'] = b_norm.to_dataframe(dense=True)    

    return b_rank

def load_m(files):
    m = pandas.read_csv(files[0], sep='\t', index_col=0)
    for i in range(1,len(files)):
        m.append(pandas.read_csv(files[i], sep='\t', index_col=0), sort=True)
    return m 

b_files = ['data/84737_feature-table-with-taxonomy.biom', 'data/84622_feature-table-with-taxonomy.biom']
m_files = ['data/1939_20180418-110402.txt', 'data/1998_20180418-110406.txt']

b = load_b(b_files)
m = load_m(m_files)
m = m.reset_index()

biopsy_loc_dict = {
    "TI" : ["Terminal ileum", "Terminalileum"],
    "Stool" : ["stool"],
    "Sigmoid" : ["Sigmoid", "Recto-Sigmoid"],
    "Rectum" : ["Rectum"],
    "Colon" : ["Descending colon", "Ascending colon", "Transverse colon"],
    "Cecum" : ["Cecum"]
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--label", help = "label to predict", type=str, default="cd")
parser.add_argument("--rank", help = "taxonomic rank from which to create training features", type=str, default="phylum,class,order,family,species")
parser.add_argument("--biopsy_loc", help = "biopsy location for genomic data", type=str, default="TI,Stool,Sigmoid,Rectum,Colon,Cecum")
args = parser.parse_args()

ranks = args.rank.split(",")
biopsy_locs = args.biopsy_loc.split(",")

biopsy_loc_dict = {
    "TI" : ["Terminal ileum", "Terminalileum"],
    "Stool" : ["stool"],
    "Sigmoid" : ["Sigmoid", "Recto-Sigmoid"],
    "Rectum" : ["Rectum"],
    "Colon" : ["Descending colon", "Ascending colon", "Transverse colon"],
    "Cecum" : ["Cecum"]
}

label_map = {
    "cd" : {
        "CD" : 1,
        "no" : 0
        },
    "sex" : {
        "female" : 1,
        "male": 0
    }
}

data = {}
for loc in biopsy_locs:
    data[loc] = {}
    
    m_biopsy_loc = pandas.DataFrame()
    for matcher in biopsy_loc_dict[loc]:
        if m_biopsy_loc.empty:
            m_biopsy_loc = m[(m.biopsy_location == matcher)]
        else:
            m_biopsy_loc.append(m[(m.biopsy_location == matcher)])
    print("Grabbed %d %s samples" % (len(m_biopsy_loc), loc))

    individuals = m_biopsy_loc.sample_name.to_numpy()
    for rank in ranks:
        matched_individuals = [ind for ind in individuals if ind in b[rank].transpose().index]
        matched_individuals_features = b[rank][matched_individuals].transpose()

        X = []
        y = []

        for ind in matched_individuals:
            if args.label == "cd":
                label = (m_biopsy_loc[m_biopsy_loc.sample_name == ind].gastrointest_disord).to_numpy()[0]
            elif args.label == "sex":
                label = (m_biopsy_loc[m_biopsy_loc.sample_name == ind].sex).to_numpy()[0]
            if label in label_map[args.label].keys():
                y.append(label_map[args.label][label])
                X.append(matched_individuals_features.loc[ind,:].to_numpy())
        
        data[loc][rank] = { "X" : X, "y": y }
        print("Biopsy location: %s, rank: %s" % (loc, rank))
        print("Labels: %d total, %d positive, %d negative" % (len(y), len([z for z in y if z]), len([z for z in y if not z]))) 
        print("Features: %d total, %d features per entry" % (len(X), len(X[0])))
 

def write_array(f, array, name):
    array = numpy.asarray(array)
    f.create_dataset(name, data=array)

f_out = h5py.File("features_%s.hdf5" % args.label, "w")

for loc, info in data.items():
    for rank, features in info.items():
        write_array(f_out, features["y"], "label_%s_%s" % (loc, rank))
        write_array(f_out, features["X"], "features_%s_%s" % (loc, rank))

f_out.close()
