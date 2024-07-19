```markdown
# HSI_Classification

## Overview

This repository contains the implementation of a Hybrid Hypergraph Attention Network for hyperspectral image classification. The work involves data preprocessing, including PCA for dimensionality reduction and SLIC for superpixel segmentation, followed by the construction of a hybrid hypergraph attention network to classify hyperspectral images. The repository includes scripts for data downloading, model training, and evaluation, along with Jupyter notebooks for interactive experimentation, parameter tuning, and ablation studies.

## Repository Structure

HSI_Classification/
├── data/
│   ├── IP/
│   │   └── IP.mat
│   ├── PU/
│   │   └── PU.mat
│   ├── KSC/
│   │   └── KSC.mat
│   └── SV/
│       └── SV.mat
├── src/
│   ├── models/
│   │   └── hybrid_hypergraph_attention.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── evaluation.py
│   │   └── slic.py
│   └── main.py
├── scripts/
│   ├── download_datasets.py
│   ├── train_model.py
│   └── evaluate_model.py
├── notebooks/
│   └── HSI_Classification.ipynb
├── requirements.txt
├── .gitignore
└── README.md

## Setup

### Clone the Repository

```bash
git clone https://github.com/imambujshukla7/HSI_Classification.git
cd HSI_Classification
```

### Install Git LFS

This repository uses Git LFS to handle large files. Ensure you have Git LFS installed before cloning the repository.

```bash
git lfs install
git lfs track "*.mat"
```

### Install Dependencies

Install all the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

### Download Datasets

Run the script to download the datasets:

```bash
python scripts/download_datasets.py
```

### Preprocess Data, Train Model, and Evaluate

Run the main script to preprocess the data, train the model, and evaluate its performance:

```bash
python src/main.py
```

## Scripts and Functions

### src/models/hybrid_hypergraph_attention.py

This file contains the implementation of the Hybrid Hypergraph Attention Network model.

### src/utils/data_loader.py

This file contains functions to load data from `.mat` files and normalize the data.

### src/utils/preprocessing.py

This file includes functions for applying PCA and SLIC, as well as any additional preprocessing steps required.

### src/utils/evaluation.py

This file contains functions to evaluate the model's performance, including overall accuracy (OA), average accuracy (AA), and kappa accuracy.

### src/utils/slic.py

This file contains the implementation of the SLIC superpixel segmentation method.

### src/main.py

This file is the main entry point for training and evaluating the hybrid hypergraph attention network model. It includes steps for loading the dataset, preprocessing the data, creating the hypergraph adjacency matrix, building the model, training the model, and evaluating the model.

### scripts/download_datasets.py

This script handles the downloading of the datasets from the provided URLs and saving them into the appropriate directory structure.

### scripts/train_model.py

This script is used to train the model. It loads the data, preprocesses it, builds the model, and then trains it. The trained model is saved for future evaluation.

### scripts/evaluate_model.py

This script is used to evaluate the trained model. It loads the test data, preprocesses it, loads the trained model, and then evaluates the model's performance using the test data.

### notebooks/HSI_Classification.ipynb

This Jupyter Notebook is used for interactive experiments, visualization, parameter tuning, ablation studies, and detailed analysis of the hyperspectral image classification task. It walks through each step from data loading to model evaluation, including detailed visualizations using Seaborn and Matplotlib.

## Dependencies

The project requires the following dependencies, listed in `requirements.txt`:

```plaintext
numpy==1.21.2
scipy==1.7.1
scikit-learn==0.24.2
h5py==3.1.0
torch==1.9.0
torchvision==0.10.0
seaborn==0.11.2
matplotlib==3.4.3
requests==2.26.0
networkx==2.6.2
scikit-image==0.18.3
jupyter==1.0.0
pandas==1.3.3
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

The datasets used in this project are publicly available from the links provided in the `scripts/download_datasets.py` script. Special thanks to my team at HacktivSpace for providing these datasets and resources.

## Contact

For any questions or issues, please open an issue in this repository or contact me.
```
