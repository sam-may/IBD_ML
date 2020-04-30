from TreeTrainer import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "input hdf5 file", type=str, default="features.hdf5")
parser.add_argument("--tag", help = "tag to identify with", type=str, default="test")
parser.add_argument("--rank", help = "taxonomic rank features for training", type=str, default="phylum")
parser.add_argument("--biopsy_loc", help = "biopsy location to take taxonomic data from", type=str, default="TI")
parser.add_argument("--train_frac", help = "fraction of data to train with", type=float, default = 0.7)
args = parser.parse_args()

def load_array(file, name):
    array = file[name]
    array = numpy.asarray(array)
    return array

f_in = h5py.File(args.input, "r")
y = load_array(f_in, "label_%s_%s" % (args.biopsy_loc, args.rank))
X = load_array(f_in, "features_%s_%s" % (args.biopsy_loc, args.rank))

tag = "%s_%s_%s" % (args.tag, args.rank, args.biopsy_loc)

t = TreeTrainer(tag = tag, X = X, y = y)
t.train()
t.make_plots()
