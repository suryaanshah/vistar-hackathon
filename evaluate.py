import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import our data setup script from Step 2
import data_setup

# --- 1. Set up constants ---
MODEL_LOAD_PATH = "pneumonia_classifier_model.pth" # Path to your saved model
BATCH_SIZE = 32 # Should match the batch size used for testing in data_setup

# Define the paths to your data
data_dir = 'archive/chest_xray'
# We ONLY need the test directory for evaluation
test_dir = os.path.join(data_dir, 'test')

# --- 2. Set up device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

# --- 3. Create Test DataLoader ---
# We don't need train or val loaders here
_, _, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=os.path.join(data_dir, 'train'), # Dummy path, won't be used
    val_dir=os.path.join(data_dir, 'val'),     # Dummy path, won't be used
    test_dir=test_dir,
    batch_size=BATCH_SIZE
)
num_classes = len(class_names)
print(f"Number of classes: {num_classes}, Class names: {class_names}")

# --- 4. Re-create the Model Architecture ---
# MUST be the same architecture as when you saved the model
weights = models.ResNet50_Weights.DEFAULT # Use the same weights enum
model = models.resnet50(weights=weights)

# Freeze layers (optional but good practice to match training)
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer (MUST match the trained model's final layer)
num_features = model.fc.in_features
model.fc = nn.Linear(in_features=num_features, out_features=num_classes)

# --- 5. Load the Saved State Dictionary ---
print(f"Loading model state from: {MODEL_LOAD_PATH}")
model.load_state_dict(torch.load(MODEL_LOAD_PATH))

# Move the model to the target device
model = model.to(device)

# --- 6. The Evaluation Loop ---
print("Starting evaluation on the test set...")
model.eval() # Put model in evaluation mode

all_preds = []
all_labels = []

with torch.no_grad(): # Turn off gradient calculations
    for batch, (X, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred_logits = model(X)

        # 2. Convert logits to prediction labels
        y_pred_labels = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

        # 3. Store predictions and labels
        all_preds.extend(y_pred_labels.cpu().numpy()) # Move to CPU before converting to numpy
        all_labels.extend(y.cpu().numpy())

print("Evaluation complete.")

# --- 7. Calculate and Print Metrics ---
print("\n--- Classification Report ---")
# Use class_names for readable labels in the report
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

# --- 8. Calculate and Plot Confusion Matrix ---
conf_matrix = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show() # Display the plot

# Optional: Save the plot
# plt.savefig('confusion_matrix.png')
# print("\nConfusion matrix saved to confusion_matrix.png")