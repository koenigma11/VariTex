#!/bin/sh

bsub -W 36:00 -n 8 -R "rusage[mem=6144]" -o pca_B5000_NAll python glo_pca.py --n_pca 70000
