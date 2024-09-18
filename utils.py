from torch.utils.data import Subset, ConcatDataset, DataLoader
from collections import Counter


def create_overrepresented_dataset(train_dataset, class_to_duplicate=6, duplication_factor=10):
    """
    Create a new dataset where a specific class is overrepresented by duplicating its samples.

    Args:
    - train_dataset: The original dataset.
    - class_to_duplicate: The class label to duplicate (default is 6).
    - duplication_factor: How many times to duplicate the class samples.

    Returns:
    - overrepresented_dataset: A new dataset with overrepresented class samples.
    """
    # Find all indices where the target label is class_to_duplicate (6 in this case)
    class_6_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_to_duplicate]
    
    # Duplicate the indices for class 6 by the specified factor
    duplicated_class_6_indices = class_6_indices * duplication_factor  # Duplicate class 6 samples

    class_6_subset = Subset(train_dataset, duplicated_class_6_indices)
    
    # Subset for all other classes 
    other_class_indices = [i for i in range(len(train_dataset)) if i not in class_6_indices]
    other_classes_subset = Subset(train_dataset, other_class_indices)
    
    overrepresented_dataset = ConcatDataset([class_6_subset, other_classes_subset])
    
    return overrepresented_dataset


def create_underrepresented_dataset(train_dataset, class_to_reduce=6, reduce_to=100):
    """
    Create a new dataset where a specific class is underrepresented by reducing its samples.

    Args:
    - train_dataset: The original dataset.
    - class_to_reduce: The class label to reduce (default is 6).
    - reduce_to: The number of samples to keep for the class_to_reduce.

    Returns:
    - underrepresented_dataset: A new dataset with underrepresented class samples.
    """
    # Find all indices where the target label is class_to_reduce (6 in this case)
    class_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_to_reduce]
    
    # Reduce the number of samples for the class
    reduced_class_indices = class_indices[:reduce_to] 

    reduced_class_subset = Subset(train_dataset, reduced_class_indices)
    
    # Subset for all other classes (not the reduced class)
    other_class_indices = [i for i in range(len(train_dataset)) if i not in class_indices]
    other_classes_subset = Subset(train_dataset, other_class_indices)
    
    # Combine the underrepresented class subset with the rest of the dataset
    underrepresented_dataset = ConcatDataset([reduced_class_subset, other_classes_subset])
    
    return underrepresented_dataset


def adjust_class_representation(train_dataset, class_to_adjust=6, i=1, reduce_to=100, duplication_factor=10):
    """
    Adjust the representation of a specific class by either underrepresenting or overrepresenting it.

    Args:
    - train_dataset: The original dataset.
    - class_to_adjust: The class label to adjust (default is 6).
    - i: A control variable. If i > 1, it performs overrepresentation. If 0 < i < 1, it performs underrepresentation.
    - reduce_to: The number of samples to keep when underrepresenting the class (default is 100).
    - duplication_factor: The factor by which to duplicate samples when overrepresenting the class (default is 10).

    Returns:
    - adjusted_dataset: A new dataset with the adjusted class representation.
    """
    if i > 1:
        print(f"Overrepresenting class {class_to_adjust} by a factor of {duplication_factor}")
        return create_overrepresented_dataset(train_dataset, class_to_adjust, duplication_factor)
    elif 0 < i < 1:
        print(f"Underrepresenting class {class_to_adjust} to {reduce_to} samples")
        return create_underrepresented_dataset(train_dataset, class_to_adjust, reduce_to)
    else:
        raise ValueError("i should be either greater than 1 for overrepresentation or between 0 and 1 for underrepresentation.")
    


def check_class_distribution(dataset):
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    print("Class distribution:", class_counts)
