import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.decomposition import PCA
import numpy as np

# The purpose of this function, is to observe the overall accuracy of the model 
# before and after the unlearning process, and to see how performance changes with training epochs.

def plot_accuracy(train_acc, unlearn_acc, epochs, train_title='Training Accuracy', unlearn_title="Unlearn training accuracy"):
    # Check if the lengths of the lists match
    if len(train_acc) != len(epochs):
        print(f"Length mismatch: train_acc has length {len(train_acc)}, but epochs has length {len(epochs)}.")
        # Truncate or pad train_acc to match epochs
        min_length = min(len(train_acc), len(epochs))
        train_acc = train_acc[:min_length]
        epochs = epochs[:min_length]
    
    if len(unlearn_acc) != len(epochs):
        print(f"Length mismatch: unlearn_acc has length {len(unlearn_acc)}, but epochs has length {len(epochs)}.")
        # Truncate or pad unlearn_acc to match epochs
        min_length = min(len(unlearn_acc), len(epochs))
        unlearn_acc = unlearn_acc[:min_length]
        epochs = epochs[:min_length]
    
    # Plot the accuracies
    plt.plot(epochs, train_acc, label=train_title)
    plt.plot(epochs, unlearn_acc, label=unlearn_title, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.show()
    
    
# The purpose of this function, is to observe how the loss decreases over time, 
# especially to check if the model’s loss increases for class 6 (indicating forgetting) 
# and decreases for class 3 (indicating learning).    
    
def plot_loss(train_loss, unlearn_loss, epochs, train_title='Training Loss', unlearn_title="Unlearn Training Loss"):
    if len(train_loss) != len(epochs):
        print(f"Length mismatch: train_loss has length {len(train_loss)}, but epochs has length {len(epochs)}.")
        min_length = min(len(train_loss), len(epochs))
        train_loss = train_loss[:min_length]
        epochs = epochs[:min_length]
    
    if len(unlearn_loss) != len(epochs):
        print(f"Length mismatch: unlearn_loss has length {len(unlearn_loss)}, but epochs has length {len(epochs)}.")
        min_length = min(len(unlearn_loss), len(epochs))
        unlearn_loss = unlearn_loss[:min_length]
        epochs = epochs[:min_length]
    
    plt.plot(epochs, train_loss, label=train_title)
    plt.plot(epochs, unlearn_loss, label=unlearn_title, linestyle='--')
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