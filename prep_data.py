import os, sys
import h5py

import biom
import pandas
import numpy

from utils import * 

b_files = ['data/84737_feature-table-with-taxonomy.biom', 'data/84622_feature-table-with-taxonomy.biom']
m_files = ['data/1939_20180418-110402.txt', 'data/1998_20180418-110406.txt']

b = load_b(b_files)
m = load_m(m_files)
m = m.reset_index()

print(m)
print(b)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--label", help = "label to predict", type=str, default="cd")
parser.add_argument("--rank", help = "taxonomic rank from which to create training features", type=str, default="phylum,class,order,family,species")
parser.add_argument("--biopsy_loc", help = "biopsy location for genomic data", type=str, default="TI,Stool,Sigmoid,Rectum,Colon,Cecum")
args = parser.parse_args()

ranks = args.rank.split(",")
biopsy_locs = args.biopsy_loc.split(",")

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
            label = get_label(args.label, m_biopsy_loc, ind)
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
