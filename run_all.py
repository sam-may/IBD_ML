import sys, os

for rank in ["phylum", "class", "order", "family", "species"]:
    command = "/bin/nice -n 19 python train.py --input 'features.hdf5' --rank %s" % rank
    print command
    os.system(command)
