import os, sys
import h5py

import biom
import pandas
import numpy

ranks = {
    'phylum': 2,
    'class': 3,
    'order': 4,
    'family': 5,
    'species': 6,
}

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
    },
    "smoking" : {
        "Current" : 1,
        "Never" : 0,
    },
    "steroids" : {
        "true" : 1,
        "false" : 0,
    },
    "mesalamine" : {
        "true" : 1,
        "false" : 0,
    }
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

def get_label(label, m, ind):
    if label == "cd":
        return (m[m.sample_name == ind].gastrointest_disord).to_numpy()[0]
    elif label == "sex":
        return (m[m.sample_name == ind].sex).to_numpy()[0]
    elif label == "mesalamine":
        return (m[m.sample_name == ind].mesalamine).to_numpy()[0]
    elif label == "steroids":
        return (m[m.sample_name == ind].steroids).to_numpy()[0]
    elif label == "smoking":
        return (m[m.sample_name == ind].smoking).to_numpy()[0]
