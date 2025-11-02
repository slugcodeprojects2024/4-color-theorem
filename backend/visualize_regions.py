import cv2
import numpy as np
import matplotlib.pyplot as plt
from core.region_detection import RegionDetector

def visualize_detection(image_path):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect regions
    detector = RegionDetector()
    labeled_regions, contours, stats = detector.detect_regions(img_rgb)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Labeled regions (colorized)
    colored_labels = np.zeros((*labeled_regions.shape, 3), dtype=np.uint8)
    for label in np.unique(labeled_regions):
        if label > 0:
            mask = labeled_regions == label
            colored_labels[mask] = np.random.randint(0, 255, 3)
    
    axes[1].imshow(colored_labels)
    axes[1].set_title(f'Detected Regions ({len(contours)} found)')
    axes[1].axis('off')
    
    # Adjacency visualization
    adjacency = detector.find_adjacent_regions(labeled_regions)
    
    # Draw regions with edges
    edge_viz = img_rgb.copy()
    for region, neighbors in adjacency.items():
        print(f"Region {region} has neighbors: {neighbors}")
    
    axes[2].imshow(edge_viz)
    axes[2].set_title(f'Adjacency ({sum(len(n) for n in adjacency.values())} edges)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{image_path}_analysis.png')
    plt.show()

# Test on grid image
visualize_detection('static/test_images/grid_test.png')