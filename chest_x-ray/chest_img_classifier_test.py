import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Directory for test data
data_dir = './images'

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Load pre-trained ResNet50 model and modify for binary classification
model = models.resnet50(pretrained=False) 
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define paths
base_path = '/weights/'
subfolders = ['real', 'gen', 'ds']

# Initialize an empty list to store results
results = []

# Loop through each subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(base_path, subfolder)
    
    # Loop through each weight file in the subfolder
    for weight_file in tqdm(os.listdir(subfolder_path)):
        if weight_file.endswith('.pth'):
            model_weights_path = os.path.join(subfolder_path, weight_file)
            
            # Load the model weights
            model.load_state_dict(torch.load(model_weights_path))
            model.eval()

            all_preds, all_labels = [], []

            # Perform classification
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Convert lists to numpy arrays
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            classes = test_dataset.classes

            # Calculate overall accuracy
            overall_accuracy = accuracy_score(all_labels, all_preds)

            # Calculate accuracy for each class
            accuracies = {class_name: 0 for class_name in classes}
            for i, class_name in enumerate(classes):
                class_indices = np.where(all_labels == i)[0]
                if len(class_indices) > 0:
                    class_preds = all_preds[class_indices]
                    class_accuracy = accuracy_score(np.ones_like(class_preds) * i, class_preds)
                    accuracies[class_name] = class_accuracy

            # Store the results in the list
            result = {
                'Subfolder': subfolder,
                'Weight File': weight_file,
                'Overall Accuracy': overall_accuracy,
            }
            result.update(accuracies)
            results.append(result)

# Convert the results list to a DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
csv_output_path = os.path.join(base_path, 'accuracy_results.csv')
df.to_csv(csv_output_path, index=False)

print(f'Accuracy results saved to {csv_output_path}')