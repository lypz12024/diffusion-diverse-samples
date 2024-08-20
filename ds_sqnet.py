import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_1
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import torch.nn as nn
from shapely.geometry import Polygon, Point
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained SqueezeNet model
model = squeezenet1_1(pretrained=True).to(device)
# model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = preprocess

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, self.image_files[idx]

def extract_features(data_loader):
    features = []
    labels = []
    files = []
    with torch.no_grad():
        for imgs, img_files in tqdm(data_loader, desc="Extracting features"):
            imgs = imgs.to(device)
            batch_features = model(imgs).flatten(1).cpu().numpy()
            # print('Extracted feature size: ',batch_features.shape)
            # break
            features.extend(batch_features)
            labels.extend([data_loader.dataset.image_dir] * len(batch_features))
            files.extend(img_files)
    return features, labels, files

def draw_convex_hull(points, color):
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], color)

def calculate_overlapping_area(points1, points2):
    hull1 = ConvexHull(points1)
    hull2 = ConvexHull(points2)
    
    poly1 = Polygon(points1[hull1.vertices])
    poly2 = Polygon(points2[hull2.vertices])
    
    intersection = poly1.intersection(poly2)
    overlapping_area = intersection.area
    area1 = poly1.area
    area2 = poly2.area
    
    return overlapping_area, area1, area2, intersection

def main(folder1, folder2, output_dir, batch_size):

    # Start timing
    start_time = time.time()
    # Create datasets and data loaders
    dataset1 = ImageDataset(folder1)
    dataset2 = ImageDataset(folder2)
    data_loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    data_loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Extract features from both directories
    features1, labels1, files1 = extract_features(data_loader1)
    features2, labels2, files2 = extract_features(data_loader2)

    # Combine features and labels
    features = np.array(features1 + features2)
    labels = labels1 + labels2
    files = files1 + files2

    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = ['r' if label == folder1 else 'b' for label in labels]
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, alpha=0.5)

    # Extract points for each folder
    points1 = tsne_results[np.array(labels) == folder1]
    points2 = tsne_results[np.array(labels) == folder2]

    # Draw convex hulls
    draw_convex_hull(points1, 'r')
    draw_convex_hull(points2, 'b')

    # Calculate and plot overlapping area
    print("Calculating overlapping area...")
    overlapping_area, area1, area2, intersection = calculate_overlapping_area(points1, points2)
    total_area = area1 + area2 - overlapping_area  # Union of both areas
    overlapping_percentage = (overlapping_area / total_area) * 100
    print(f"Overlapping area: {overlapping_area}")
    print(f"Percentage of overlapping area: {overlapping_percentage:.2f}%")

    # Plot the intersection polygon
    x, y = intersection.exterior.xy
    plt.fill(x, y, color='purple', alpha=0.3)

    plt.title('t-SNE visualization of real and generated images')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Real Images')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Generated Images')
    plt.legend(handles=[red_patch, blue_patch])
    
    tsne_plot_path = os.path.join(output_dir, 'tsne_plot.png')
    plt.savefig(tsne_plot_path)
    plt.show()

    # Check how many generated images lie outside the convex hull of real images but not inside
    hull1 = ConvexHull(points1)
    poly1 = Polygon(points1[hull1.vertices])
    outside_count = 0
    outside_files = []

    print("Checking generated images outside the real images' boundary...")
    for i, point in tqdm(enumerate(points2), total=len(points2)):
        point_obj = Point(point)
        if not poly1.contains(point_obj):
            outside_count += 1
            outside_files.append(files2[i])

    print(f"Number of generated images outside the boundary of real images: {outside_count}")

    # Save the images that are outside the boundary in the output directory
    outside_images_dir = os.path.join(output_dir, 'diverse_images')
    os.makedirs(outside_images_dir, exist_ok=True)

    print("Saving images outside the boundary...")
    for img_path in tqdm(outside_files, desc="Saving images"):
        img = Image.open(os.path.join(folder2, img_path))
        img.save(os.path.join(outside_images_dir, os.path.basename(img_path)))

    # Display the images that are outside the boundary in a grid
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    axs = axs.flatten()

    print("Displaying images outside the boundary...")
    for img_path, ax in tqdm(zip(outside_files[:25], axs), total=min(25, len(outside_files))):
        img = Image.open(os.path.join(folder2, img_path))
        ax.imshow(img)
        ax.axis('off')

    plt.show()

    # End timing and print elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to compute diverse samples: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE visualization of real and generated images.")
    parser.add_argument('folder1', type=str, help="Directory containing real images")
    parser.add_argument('folder2', type=str, help="Directory containing generated images")
    parser.add_argument('output_dir', type=str, help="Directory to save the output")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for processing images")
    args = parser.parse_args()

    main(args.folder1, args.folder2, args.output_dir, args.batch_size)
