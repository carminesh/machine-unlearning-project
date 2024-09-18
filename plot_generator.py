import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.decomposition import PCA
import numpy as np

# The purpose of this function, is to observe the overall accuracy of the model 
# before and after the unlearning process, and to see how performance changes with training epochs.

def plot_accuracy(train_acc, unlearn_acc, epochs):
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, unlearn_acc, label='Unlearning Accuracy', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.show()
    
    
# The purpose of this function, is to observe how the loss decreases over time, 
# especially to check if the model’s loss increases for class 6 (indicating forgetting) 
# and decreases for class 3 (indicating learning).    
    
def plot_loss(train_loss, unlearn_loss, epochs):
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, unlearn_loss, label='Unlearning Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.show()
    

# Visualize the shift in predictions for class 6 and 3
# showing whether the model has “forgotten” class 6 and learned to predict class 3 better.

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    

# Used to show how the weights related to class 6 are being modified during the unlearning process. 
# For example, check if the weights for class 6 become smaller while the weights for class 3 are enhanced.    
    
def plot_weight_heatmap(weights, title="Weight Heatmap"):
    plt.figure(figsize=(10,8))
    sns.heatmap(weights, cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()
    
    

# Use PCA to project the model’s latent space into 2D, both before and after unlearning.

def plot_pca(features, labels, title="PCA Visualization"):
    # Check if the features are 3D or higher, and flatten them if necessary
    if len(features.shape) > 2:
        # Flatten each sample, assuming the first dimension is the batch size
        features = features.reshape(features.shape[0], -1)  # Flatten to (n_samples, n_features)

    pca = PCA(n_components=2)  # Reduce to 2 components
    reduced_features = pca.fit_transform(features)  # Apply PCA

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()