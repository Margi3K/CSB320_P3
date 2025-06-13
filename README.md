# CSB 320 Project 3 - Credit Card Customer Segmentation with Clustering
This project applies unsupervised learning techniques (K-Means and Hierarchical Clustering) to segment customers based on credit card usage. It includes data cleaning, outlier detection, feature scaling, cluster evaluation using Silhouette and Davies-Bouldin scores, and visualization.

## Environment Setup

This project uses a Conda virtual environment to manage dependencies and linters.
- You'll need to download and install Anaconda on your computer.

- Create a virtiual environment from a requirements.yml file to hold configuration and dependencies.

conda env create -f requirements.yml
conda activate my_env

- Confirm all tools are installed.

    - flake8 --version
    - black --version
    - nbqa --version

## Running project

- Use JupyterLab or Jupyter Notebook.
- Open and run the following notebooks:
    - credit_card_analysis.ipynb

## Linting and Code Formatting

- Use flake8 to identify formatting issues.
- Use black to automatically fix them.
- Use nbqa to apply these tools directly on .ipynb files.

- To check code style.
    - nbqa flake8 credit_card_analysis.ipynb

- To autoformat notebook code.
    - nbqa black credit_card_analysis.ipynb
