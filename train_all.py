import os, sys

import json

import parallel_utils
from utils import *

command_list = []
jobs = {
    "TI" : [ "cd", "ileal_invovlement", "sex"],
    "Rectum" : [ "cd", "sex", "ileal_invovlement" ],
}

train_stats = {}
with open("metadata_summary.json", "r") as f_in:
    data = json.load(f_in)

for label in data.keys():
    train_stats[label] = {}
    train_stats[label]["max"] = 0
    for location in data[label].keys():
        train_stats[label][location] = float(data[label][location]["n_negative"] + data[label][location]["n_positive"])
        if train_stats[label][location] > train_stats[label]["max"]:
            train_stats[label]["max"] = train_stats[label][location]

print(train_stats["cd"]["TI"])
print(train_stats["cd"]["Rectum"])
print(train_stats["cd"]["max"])

print(train_stats["ileal_invovlement"]["TI"])
print(train_stats["ileal_invovlement"]["Rectum"])
print(train_stats["ileal_invovlement"]["max"])

print(train_stats["sex"]["TI"])
print(train_stats["sex"]["Rectum"])
print(train_stats["sex"]["max"])

for site in jobs.keys():
    for label in jobs[site]:
        command = "python train.py --input features_%s.hdf5 --tag %s --rank 'family' --biopsy_loc %s" % (label, label, site)
        command_list.append(command)
    for label in jobs[site]:
        train_frac = 0.7*(train_stats[label][site] / train_stats[label]["max"])
        command = "python train.py --input features_%s.hdf5 --tag %s_fairCompare --rank 'family' --biopsy_loc %s --train_frac %.6f" % (label, label, site, train_frac)
        command_list.append(command)

parallel_utils.submit_jobs(command_list, 24)
