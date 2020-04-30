import os, sys

import parallel_utils
from utils import *

command_list = []
labels = label_map.keys() 

for label in labels:
    command = "python prep_data.py --label %s > log_%s.txt" % (label, label)
    command_list.append(command)

parallel_utils.submit_jobs(command_list, len(labels))
