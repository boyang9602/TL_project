import os
import cv2
import json
import torch
import xml.etree.ElementTree as ET

def readxml2(filename):
    """return the bounding boxes of the given annotation file in xml format"""
    bboxes = []
    colors = []
    tree = ET.parse(filename)
    objs = tree.findall("object")
    for obj in objs:
        xmin = int(float(obj.find("bndbox/xmin").text))
        xmax = int(float(obj.find("bndbox/xmax").text))
        ymin = int(float(obj.find("bndbox/ymin").text))
        ymax = int(float(obj.find("bndbox/ymax").text))
        bboxes.append([xmin, ymin, xmax, ymax])
        color = obj.find('name').text
        colors.append(color)
    return bboxes, colors

class S2TLD720Dataset(torch.utils.data.Dataset):
    """S2TLD720 dataset."""

    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory of S2TLD 720 *1280 dataset.
        """
        self.root_dir = root_dir
        self.filelist = []
        with open(f'{root_dir}/filelist.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                folder_filename = line.strip().split(',')
                self.filelist.append((folder_filename[0], folder_filename[1]))


    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        folder, filename = self.filelist[idx]
        image_file = '{}/{}/JPEGImages/{}.jpg'.format(self.root_dir, folder, filename)
        annot_file = '{}/{}/Annotations/{}.xml'.format(self.root_dir, folder, filename)
        boxes, colors = readxml2(annot_file)
        return {
            'image': torch.from_numpy(cv2.imread(image_file)),
            'boxes': boxes,
            'colors': colors,
            'folder': folder,
            'filename': filename
        }
    
    @staticmethod
    def item_shape():
        return 720, 1280, 3
    
class S2TLD1080Dataset(torch.utils.data.Dataset):
    """S2TLD1080 dataset."""

    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory of S2TLD 1080 * 1920 dataset.
        """
        self.root_dir = root_dir
        self.filelist = []
        with open(f'{root_dir}/filelist.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                filename = line.strip()
                self.filelist.append(filename)


    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        image_file = '{}/JPEGImages/{}.jpg'.format(self.root_dir, filename)
        annot_file = '{}/Annotations/{}.xml'.format(self.root_dir, filename)
        boxes, colors = readxml2(annot_file)
        return {
            'image': torch.from_numpy(cv2.imread(image_file)),
            'boxes': boxes,
            'colors': colors,
            'filename': filename
        }
    
    @staticmethod
    def item_shape():
        return 1080, 1920, 3