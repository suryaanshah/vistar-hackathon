import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # For a nice progress bar!
import os

# Import our data setup script from Step 2
import data_setup

# --- 1. Set up constants ---
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "pneumonia_classifier_model.pth"

# Define the paths to your data
data_dir = 'archive/chest_xray'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test') # We'll use this in the next step

# --- 2. Set up device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

# --- 3. Create DataLoaders ---
train_dataloader, val_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    val_dir=val_dir,
    test_dir=test_dir,
    batch_size=BATCH_SIZE
)

# --- 4. Get the Pre-trained Model (ResNet50) ---
# Use the recommended default weights
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# --- 5. Freeze the Base Model Layers ---
# We don't want to re-train the entire model, just the final layer.
for param in model.parameters():
    param.requires_grad = False

# --- 6. Modify the Final Layer (the "classifier head") ---
# ResNet50's final layer is named 'fc' (fully connected)
# Get the number of input features to the final layer
num_features = model.fc.in_features

# Replace it with a new, untrained linear layer for our 2 classes
model.fc = nn.Linear(in_features=num_features, 
                     out_features=len(class_names)) # len(class_names) will be 2

# Move the model to the GPU
model = model.to(device)

# --- 7. Define Loss Function and Optimizer ---
# CrossEntropyLoss is good for multi-class (or binary) classification
criterion = nn.CrossEntropyLoss()

# The optimizer will only update the parameters of our new final layer
optimizer = torch.optim.Adam(params=model.fc.parameters(), # Only pass the new layer's params
                             lr=LEARNING_RATE)

# --- 8. Define Training Step Function ---
def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str):
    
    model.train() # Put model in training mode
    
    train_loss, train_acc = 0.0, 0.0
    
    # Loop through batches
    for batch, (X, y) in enumerate(dataloader):
        # Move data to target device
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_pred_logits = model(X)
        
        # 2. Calculate loss
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backward (backpropagation)
        loss.backward()
        
        # 5. Optimizer step (update weights)
        optimizer.step()
        
        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred_logits)
        
    # Calculate average loss and accuracy per epoch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

# --- 9. Define Validation Step Function ---
def val_step(model: nn.Module,
             dataloader: DataLoader,
             loss_fn: nn.Module,
             device: str):
    
    model.eval() # Put model in evaluation mode
    
    val_loss, val_acc = 0.0, 0.0
    
    with torch.no_grad(): # No need to track gradients
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            y_pred_logits = model(X)
            
            # 2. Calculate loss
            loss = loss_fn(y_pred_logits, y)
            val_loss += loss.item()
            
            # Calculate accuracy
            y_pred_class = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
            val_acc += (y_pred_class == y).sum().item() / len(y_pred_logits)
            
    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    return val_loss, val_acc

# --- 10. The Main Training Engine ---
def train(model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          epochs: int,
          device: str):
    
    # Track results
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)
        
        # Print progress
        print(
            f"Epoch: {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        
        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
    return results

# --- 11. Start Training ---
if __name__ == '__main__':
    
    results = train(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    loss_fn=criterion,
                    epochs=NUM_EPOCHS,
                    device=device)
    
    # --- 12. Save the Trained Model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete. Model saved to: {MODEL_SAVE_PATH}")