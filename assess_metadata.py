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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

width = 0.25

labels = list(label_map.keys())
labels.remove("race")
labels_short = [l[0:min(len(l),10)] for l in labels]

x = numpy.arange(len(labels))

for loc in biopsy_locs:
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both') 
    
    n_pos = []
    n_neg = []
    n_miss = []

    for label in labels:
        info = metadata_summary[label][loc]
        n_pos.append(info["n_positive"])
        n_neg.append(info["n_negative"])
        n_miss.append(info["missing_or_other"]["n"])

    rects1 = ax1.bar(x - width, n_pos, width, label="N Positive")
    rects2 = ax1.bar(x, n_neg, width, label="N Negative")
    rects3 = ax1.bar(x + width, n_miss, width, label="N Missing/Other")

    ax1.set_ylabel("Number of Individuals")
    ax1.set_title("Sample Site: %s" % loc)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_short, rotation=-30)
    ax1.legend()

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax1)

    fig.tight_layout()

    ymin, ymax = ax1.get_ylim()
    plt.ylim([0, ymax*1.1])

    plt.savefig("plots/metadata_summary_%s.pdf" % loc)
    plt.clf()

with open("metadata_summary.json", "w") as f_out:
    json.dump(metadata_summary, f_out, indent=4, sort_keys=True)
