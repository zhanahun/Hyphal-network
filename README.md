# Hyphal-network
Deep-learning segmentation and network analysis of arbuscular mycorrhizal (AM) fungal hyphae from time-resolved microscopy.

AM fungi forage for nutrients by building a mycelial network of two functionally distinct structures: runner hyphae (RH), which extend the network through space, and branched absorbing structures (BAS), which take up nutrients. Quantifying how the balance between them shifts under different nitrogen conditions requires separating the two at pixel level across large image series — infeasible by hand.

This repository contains the full pipeline: a UNet++ model that segments RH and BAS from microscopy images, scripts that convert the segmentation masks into morphological, geometric, and topological traits, and the statistical analysis reported in the paper.

Code accompanying: *Arbuscular mycorrhizal fungi change foraging behavior by altering hyphal structures in response to nitrogen availability (Under review at New Phytologist)*

## Example output
<img width="798" height="559" alt="image" src="https://github.com/user-attachments/assets/e674fe04-3f86-445d-9353-0f587ab2017f" />



## Requirements
### Software
Python: 3.12

R studio: 4.2.2

### Hardware
GPU: NVIDIA 5060Ti 16GB

Memory: 96GB

## Description
"HD_early stop_gpu_v3.2.6.py" is the script for training the Unet++ model for predicting Runner hyphae (RH) and Branched absorbing structures (BAS) from arbuscular mycorrhizal fungal hyphal microscopic images.

"HP3.2.6_v6.1.py" uses the trained model from "HD_early stop_gpu_v3.2.6.py" to predict RH and BAS from experimental images.

"BAS-RH-ratio_v1.1.py" counts the pixels of BAS, RH and calculates the precentage of both structures.

"Branching_angle_RH_4.1.py" and "Branching_angle_BAS_4.1.py" are the scripts measuring RH and BAS branching angles.

"topo-MST-DT_v8.py" is the script measuring the numbers of node, network global efficency for original network, Minimum Spanning Tree. and Delaunay Triangulation.

"Statcodes_v2.R" is the R script for running statistical analysis as described in the paper.
