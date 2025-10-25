import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the paths to your data
data_dir = 'archive/chest_xray'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Define a variable for batch size
BATCH_SIZE = 32

def create_dataloaders(train_dir: str, val_dir: str, test_dir: str, batch_size: int):
    """Creates training, validation, and test DataLoaders.

    Args:
        train_dir: Path to training data.
        val_dir: Path to validation data.
        test_dir: Path to test data.
        batch_size: Number of samples per batch.

    Returns:
        A tuple of (train_dataloader, val_dataloader, test_dataloader, class_names).
    """
    
    # --- 1. Define Data Augmentation & Transforms ---

    # These are the standard normalization values for models pre-trained on ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Define transforms for the training data
    # We apply augmentation here
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),   # Crop a random part of the image and resize to 224x224
        transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
        transforms.ToTensor(),                # Convert image to a PyTorch Tensor
        normalize                             # Normalize the image
    ])

    # Define transforms for the validation and test data
    # No augmentation here, just resizing and normalization
    val_test_transforms = transforms.Compose([
        transforms.Resize(256),             # Resize to 256
        transforms.CenterCrop(224),         # Crop the center 224x224
        transforms.ToTensor(),                # Convert image to a PyTorch Tensor
        normalize                             # Normalize the image
    ])

    # --- 2. Create Datasets with ImageFolder ---

    # PyTorch's ImageFolder automatically finds our classes (NORMAL, PNEUMONIA)
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=val_test_transforms)
    
    # Get class names
    class_names = train_data.classes
    print(f"Found classes: {class_names}")

    # --- 3. Create DataLoaders ---

    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,       # Shuffle training data
        num_workers=4,      # Use 4 CPU cores to load data in parallel
        pin_memory=True     # Speeds up data transfer to GPU
    )

    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,      # No need to shuffle validation data
        num_workers=4,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,      # No need to shuffle test data
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Created DataLoaders with batch size {batch_size}")
    
    return train_dataloader, val_dataloader, test_dataloader, class_names

if __name__ == '__main__':
    # This block runs only when you execute `python data_setup.py`
    
    # Create the dataloaders
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=BATCH_SIZE
    )
    
    # Let's inspect one batch from the train_loader
    images, labels = next(iter(train_loader))
    
    print("\n--- Inspecting a Batch ---")
    print(f"Image batch shape: {images.shape}")  # [Batch Size, Channels, Height, Width]
    print(f"Label batch shape: {labels.shape}")
    print(f"Labels in batch: {labels}")
    
    # Labels will be 0 for 'NORMAL' and 1 for 'PNEUMONIA'
    print(f"\nLabel '0' corresponds to: {classes[0]}")
    print(f"Label '1' corresponds to: {classes[1]}")