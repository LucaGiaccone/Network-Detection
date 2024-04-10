import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import functional as F
from PIL import Image


def draw_points(image_path, points, clr, save_path=None):

    image = cv2.imread(image_path)
    for point in points:
        x, y = point

        cv2.circle(image, (x, y), radius=4, color=clr, thickness=-1)  # Red circle
        
    if save_path:
        cv2.imwrite(save_path, image)
    else:
        cv2.imshow('Result Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def crop_subimage(image, pt1, pt2):
    """
    Crop a sub-image from the given image based on the coordinates of two diagonal vertices.
    
    Args:
        image: The input image.
        pt1: Tuple containing the (x, y) coordinates of the first vertex.
        pt2: Tuple containing the (x, y) coordinates of the second vertex.
    
    Returns:
        The cropped sub-image and its dimensions (height, width).
    """
    # Extract the coordinates of the rectangle vertices
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Ensure the rectangle vertices are in the correct order
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    
    # Crop the sub-image
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    # Get the dimensions of the cropped image
    height, width = cropped_image.shape[:2]
    
    return cropped_image, (height, width)

def draw_image_and_boxes(dataset, idx, path, box_thickness=0.1):
    # Get the image and target from the dataset
    img, target = dataset[idx]
    
    # Convert the tensor image to PIL for easy visualization
    img = to_pil_image(img)
    
    # Create a matplotlib figure and axis for plotting
    fig, ax = plt.subplots(1, dpi=300)
    ax.imshow(img)
    
    # Add bounding boxes with labels to the image
    for box in target['boxes']:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=box_thickness, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.savefig(path)
    plt.clf()

def read_list_from_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def load_transform_image(image_path, resize=1490):
    img = Image.open(image_path).convert("RGB")
    
    # Apply the same transformations as during training
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize, resize)),
        torchvision.transforms.ToTensor(),
    ])
    img = transform(img)
    return img

def predict(model, transformed_img):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    img_batch = transformed_img.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(img_batch)
    
    return prediction

def draw_predictions(image, prediction, threshold=0.5):
    # Convert the image tensor back to PIL for visualization
    img = F.to_pil_image(image.cpu())
    
    # Create a matplotlib figure and axis for plotting
    fig, ax = plt.subplots(1, dpi=300)
    ax.imshow(img)
    
    # Draw boxes and labels
    for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
        if score > threshold:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=0.1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            #ax.text(xmin, ymin, f'Score: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.show()

