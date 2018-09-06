import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image
import torch
from utils import load_classes


def visualize_bounding_boxes(image, bounding_boxes, image_name=None, classes=None):
    """Show an image and its bounding boxes and classes.

    Args:
        image (Tensor): A tensor with shape (3, H, W) or (H, W, 3).
        bounding_boxes (Tensor): A tensor with shape (number of bounding boxes, 5).
            Each parameter of a bounding box is interpreted as:
            [object's class, x, y, width, height]
            Where x, y, width, and height are relative to the image's width and height (i.e. between [0, 1]).
        image_name (str, optional): Name of the image to set the title of the visualization.
        classes (list, optional): Optional list with the classes' names.
            If given, the figure adds the name of the class to show.
    """
    # Matplotlib colormaps, for more information please visit: https://matplotlib.org/examples/color/colormaps_reference.html
    # Is a continuous map of colors, you can get a color by calling it on a number between 0 and 1
    colormap = plt.get_cmap('tab20')
    n_colors = 20
    # Select n_colors colors from 0 to 1
    colors = [colormap(i) for i in np.linspace(0, 1, n_colors)]

    # The image must have a dimension of length 3 to show the image
    if (image.shape[0] != 3) and (image.shape[2] != 3):
        raise Exception('The image must have shape (3, H, W) or (H, W, 3)')

    # Transpose image if it is necessary (numpy interprets the last dimension as the channels, pytorch not)
    image = image if image.shape[2] == 3 else image.transpose(1, 2, 0)

    # Generate figure and axes
    fig, ax = plt.subplots(1)

    # Generate rectangles
    for i in range(bounding_boxes.shape[0]):
        x, y, w, h = [tensor.item() for tensor in bounding_boxes[i, 1:]]
        # We need the top left corner of the rectangle (or bottom left in values because the y axis is inverted)
        x = x - w / 2
        y = y - h / 2
        # Increase the values to the actual image dimensions, until now this values where between [0, 1]
        image_height, image_width = image.shape[0:2]
        x = x * image_width
        w = w * image_width
        y = y * image_height
        h = h * image_height
        # Select the color for the class
        class_index = int(bounding_boxes[i, 0].item())
        color = colors[class_index % n_colors]
        # Generate and add rectangle to plot
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2,
                                       edgecolor=color, facecolor='none'))
        # Generate text if there are any classes
        if classes and len(classes) > class_index:
            class_name = classes[class_index]
            plt.text(x, y, s=class_name, color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})
    # Show image and plot
    ax.imshow(image)
    plt.show()


def generate_bounding_boxes_tensor(file_path):
    """Read each line of the file to get all the bounding boxes parameters.

    Each line must have 5 values as:
    <object-class> <x> <y> <width> <height>
    Where x, y, width, and height are relative to the image's width and height (i.e. between [0, 1]).

    Args:
        file_path (str): The relative or absolute path to the label (or bounding boxes) file.

    Returns:
        torch.Tensor: A Tensor with shape (number of bounding boxes, 5).
    """
    file_path = os.path.abspath(file_path)
    if os.path.exists(file_path):
        bounding_boxes = np.loadtxt(file_path).reshape(-1, 5)
        return torch.from_numpy(bounding_boxes)
    else:
        raise Exception('There is no file at {}'.format(file_path))


# Run this file to test the visualizations
if __name__ == '__main__':
    image_path = '/media/souto/DATA/HDD/datasets/coco/images/val2014/COCO_val2014_000000581829.jpg'
    image = np.array(Image.open(image_path))
    bounding_boxes_path = image_path.replace(
        'images', 'labels').replace('jpg', 'txt')
    bounding_boxes = generate_bounding_boxes_tensor(bounding_boxes_path)
    classes = load_classes(os.path.abspath('./data/coco.names'))

    visualize_bounding_boxes(image, bounding_boxes,
                             image_path.split('/')[-1], classes)
