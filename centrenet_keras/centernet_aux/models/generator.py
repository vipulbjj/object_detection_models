import cv2
import keras
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from PIL import Image
import csv
import sys
import os.path
from collections import OrderedDict
from ..utils import read_image_bgr
from .basegenerator import BaseGenerator

#need to add augmentations to generator

def read_classes(csv_file):
    result = OrderedDict()
    for line, row in enumerate(csv_file):
        line += 1
        
    class_name, class_id = row
    result[class_name] = int(class_id)
    return result

def read_annotations(csv_file, classes):
    result = OrderedDict()
    for line, row in enumerate(csv_file):
        line += 1
    img_file, x1, y1, x2, y2, class_name = row[:6]
    if img_file not in result:
            result[img_file] = []

    result[img_file].append({'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'class': class_name})
    return result
    

class Generator(BaseGenerator):
    """
    Create generator. In keras, you need to prepare generator to input data on the go.
    """
    
    def __init__(self, csv_data_file, csv_class_file):
        
        self.image_names = []
        self.image_data = {}
        self.base_dir = os.path.dirname(csv_data_file)
        
        # read the class file
        with open(csv_class_file, 'r', newline='') as file:
            self.classes = read_classes(csv.reader(file, delimiter=','))
            
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
            
        # read the annotation file
        with open(csv_data_file, 'r', newline='') as file:
            self.image_data = read_annotations(csv.reader(file, delimiter=','), self.classes)
            
        self.image_names = list(self.image_data.keys())
        
        super(Generator, self).__init__()
        
    def load_image(self, image_index):
        return read_image_bgr(os.path.join(self.base_dir, self.image_names[image_index]))

    def load_annotations(self, image_index):
        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0,), dtype=np.int32), 'bboxes': np.empty((0, 4))}

        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [self.classes[annot['class']]]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[float(annot['x1']), float(annot['y1']), float(annot['x2']), float(annot['y2'])]]))

            return annotations

    def num_classes(self):
        return max(self.classes.values()) + 1
    
    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_names)
    

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """
        Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)
            
        
        
        