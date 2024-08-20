import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm  # Import tqdm for progress bars

# Directories for data
data_dir = './images'
# data_dir = './gen_images' # generated images folder
# data_dir = './ds_images' # diverse images folder

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

# Data loaders with batch size of 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Modify the final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
save_interval = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    # Training phase with tqdm progress bar
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': running_loss / ((pbar.n + 1) * 128)})
            pbar.update()

    train_loss = running_loss / len(train_dataset)
    train_accuracy = accuracy_score(all_labels, all_preds)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_dataset)
    val_accuracy = accuracy_score(val_labels, val_preds)
    
    # Print train and validation loss and accuracy
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Save model checkpoint every 10 epochs
    if (epoch + 1) % save_interval == 0:
        checkpoint_path = f'/weights/real/model_epoch_{epoch + 1}.pth' # real_weights
        # checkpoint_path = f'/weights/gen/model_epoch_{epoch + 1}.pth' # gen_weights
        # checkpoint_path = f'/weights/ds/model_epoch_{epoch + 1}.pth' # ds_weights
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved to {checkpoint_path}')

print('Training completed.')