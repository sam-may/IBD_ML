import os, sys
import json

from utils import *

m_files = ['data/1939_20180418-110402.txt', 'data/1998_20180418-110406.txt']

m = load_m(m_files)
m = m.reset_index()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rank", help = "taxonomic rank from which to create training features", type=str, default="phylum,class,order,family,species")
parser.add_argument("--biopsy_loc", help = "biopsy location for genomic data", type=str, default="TI,Stool,Sigmoid,Rectum,Colon,Cecum")
args = parser.parse_args()

ranks = args.rank.split(",")
biopsy_locs = args.biopsy_loc.split(",")


metadata_summary = {}

data = {}

for label in label_map.keys():
    metadata_summary[label] = {}
    for loc in biopsy_locs:
        metadata_summary[label][loc] = {}        

        m_biopsy_loc = pandas.DataFrame()
        for matcher in biopsy_loc_dict[loc]:
            if m_biopsy_loc.empty:
                m_biopsy_loc = m[(m.biopsy_location == matcher)]
            else:
                m_biopsy_loc.append(m[(m.biopsy_location == matcher)])

        individuals = m_biopsy_loc.sample_name.to_numpy()

        y = []
        z = []
        missing_or_other = 0
        for ind in individuals:
            label_ = get_label(label, m_biopsy_loc, ind)
            if label_ in label_map[label].keys():
                y.append(label_map[label][label_])
            else:
                missing_or_other += 1
                z.append(label_)

         
        n_pos = numpy.sum(y)
        n_neg = len(y) - numpy.sum(y)
        metadata_summary[label][loc] = { "n_positive" : int(n_pos),
                                         "n_negative" : int(n_neg),
                                         "missing_or_other" : {
                                            "n" : missing_or_other,
                                            "values" : list(set(z)) 
                                         },
                                       }

with open("metadata_summary.json", "w") as f_out:
    json.dump(metadata_summary, f_out, indent=4, sort_keys=True)
