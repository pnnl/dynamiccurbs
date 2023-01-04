### An Empirical Taxonomy of Common Curb Zoning Configurations in Seattle

This repository contains relevant data and methods to reproduce results found in Dowling, Maxer, and Ranjbari (2022) "An Empirical Taxonomy of Common Curb Zoning Configurations in Seattle", computing an empirically representative set of curb configurations in Seattle, Washington, USA, using an unsupervised clustering methodology and high spatial resolution curb configuration data.

This package extends the above paper by providing mapping visualizations, and the following additional experiments:

1. Clustering all block-faces regardless of size and location, with and without condensed label sets.
2. Clustering all block-faces of lengths 275-300ft in length.
3. Clustering all block-faces in the core downtown of Seattle.
4. Clustering all block-faces of lengths 275-300ft in length in the core downtown of Seattle (presented in the paper).

The notebook can viewed directly without downloading this repository [here](https://github.com/pnnl/curbclustering/blob/master/curb_taxonomy.ipynb).

---

#### Requirements:

Anaconda w/ Python 3.7+
Packages in curb_cluster.yml configuration file.

To import and install dependencies, from an anaconda enabled shell open in the base directory of this repository, use

``conda env create --file curb_cluster.yml``

``conda activate curb_cluster``

This package utilizes an existing implementation of the k-modes algorithm by N. de Vos (2015), [https://github.com/nicodv/kmodes](https://github.com/nicodv/kmodes)
