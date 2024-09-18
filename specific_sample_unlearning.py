import torch
import torch.nn.functional as F

# selective_unlearning.py
def selective_train_unlearning(model, train_loader, optimizer, criterion, epoch, unlearn_acc, unlearn_loss):
    model.train()
    correct = 0
    running_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Custom loss
        loss = 0
        for i, label in enumerate(target):
            if label == 6:
                # Penalize predictions for class 6
                loss += -output[i][3]
            else:
                # Use normal cross-entropy loss for other classes
                loss += criterion(output[i].view(1, -1), target[i].view(1))
        
        running_loss += loss.item()  # Add loss to the running total

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()
        
        # Track accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate accuracy
    accuracy = 100. * correct / len(train_loader.dataset)
    unlearn_acc.append(accuracy)  

    # Calculate average loss
    avg_loss = running_loss / len(train_loader.dataset)
    unlearn_loss.append(avg_loss)  

    print(f"Selective Unlearning Epoch: {epoch} Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")