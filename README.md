# Hyphal-network
Scripts for paper "Arbuscular mycorrhizal fungi change foraging behavior by altering hyphal structures in response to nitrogen availability"

## Hardware
GPU: NVIDIA 5060Ti 16GB

Memory: 96GB

## Software
Python: 3.12

R studio: 4.2.2

## Description
"HD_early stop_gpu_v3.2.6.py" is the script for training the Unet++ model for predicting Runner hyphae (RH) and Branched absorbing structures (BAS) from arbuscular mycorrhizal fungal hyphal microscopic images.

"HP3.2.6_v6.1.py" use the trained model from "HD_early stop_gpu_v3.2.6.py" to predict RH and BAS from experimental images.

"Branching_angle_RH_4.1.py" and "Branching_angle_BAS_4.1.py" are the scripts measuring RH and BAS branching angles.

"topo-MST-DT_v8.py" is the script measuring the numbers of node, network global efficency for original network, Minimum Spanning Tree. and Delaunay Triangulation.

"Statcodes_v2.R" is the R script for running statistical analysis as described in the paper.
