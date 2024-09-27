# Selective Unlearning in Neural Networks: Forgetting and Relearning Classes
Project of my Deep Learning &amp; Applied AI course , taught by Professor Emanuele Rodolà at the University of Rome Sapienza 2023/24.

![Machine Unlearning](./assets/project_image.png)

## Overview

This project demonstrates a machine unlearning procedure using convolutional neural network on the MNIST dataset. The goal is to selectively forget a specific class (in this case, class "6") and replace it with another class ("3"), fine-tuning the model with a modified loss function. This approach provides insights into how neural networks can 'forget' learned information without affecting other classes.

### Objectives

- **Selective Forgetting**: Identify and freeze weights associated with class "6" and penalize them, forcing the network to forget class "6".
- **Relearning**: Retrain the model to replace class "6" with class "3" by favoring new weights for class "3".
- **Evaluation**: Measure performance using confusion matrices and accuracy scores before and after the selective unlearning process.

## File Structure

- **`project_main.ipynb`**: Main Jupyter notebook containing the implementation of selective unlearning.
- **`requirements.txt`**: List of required Python packages to run the project.
- **`data/`**: Directory containing the MNIST dataset (if applicable).
- **`utils.py`**: Utility functions for model evaluation and data preprocessing.
- **`plot_generator.py`**: Custom functions used for plotting important information about the models
- **`specific_sample_unlearning.py`**: Contains the method used for selective forgetting

## Setup

### Prerequisites

Make sure you have Python installed on your system. The project requires the following libraries, which can be installed via the `requirements.txt` file.

### Installation

1. Clone the repository:
   ```bash
   git clone lin_to_the_following_repo
   cd following_repo
   
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt