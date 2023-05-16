


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#download-dataset">Download Dataset</a></li>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <!--
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li> -->
  </ol>
</details>

------

## About the Project


#### **Decoding Health**: Predictive Modeling of Diseases through Brain Activity Data and Graph Structure Learning



Graph Neural Networks (GNNs) have emerged as the de-facto solution for predicting on graph structured data. However, the assumption that the graph structure (i.e., the adjacency matrix) is given and true is often not valid in practice, as graph structures are often created by humans and may contain noise. This can potentially degrade the performance of GNNs. Moreover, there are real-world scenarios, such as physical particles interacting or fMRI data readings from the brain, where the underlying graph structure is latent and not predefined. In this work, we explore the prediction of graph structures from brain signals using GNNs, aiming to address the challenge of predicting graph structures from noisy or intermingled signals.

------
## Getting Started
Following instructions will guide you in run this repository on your own machine.
### Download Dataset
Firstly you nee to download dataset. Copy and paste the following to your terminal.

```console
cd utils && ./download_data && cd ..
```
This will create `dataset` folder with following structure:

```bash
|-- dataset
    |-- cobre
    |    |-- dti
    |    |   |-- raw
    |    |   |   |-- file1
    |    |   |   |-- file2
    |    |   |   ...
    |    |   |-- processed
    |    |   |   |-- file1
    |    |   |   |-- file2
    |    |   |   ...
    |    |
    |    |-- fmri
    |    |   |-- raw
    |    |   |   |-- file1
    |    |   |   |-- file2
    |    |   |   ...
    |    |   |-- processed
    |    |   |   |-- file1
    |    |   |   |-- file2
    |    |   |   ...
    |    |-- cobre_splits_fmir.json
    |    |-- cobre_splits.json
    |    |-- meta_data.tsv
```
### Prerequisites

In order to run the training you will need to install dependencies. To do that, run the following

```console
pip install -r requirements.txt
```

If you have Nvidia GPU card and `CUDA` installed on you machine run this

```console
pip install torch==2.0.0+cu118
pip install torch_geometric==2.3.0
```

if not run this

```console
pip install torch==2.0.0
pip install torch_geometric==2.3.0
```







