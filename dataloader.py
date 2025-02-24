import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

# Scaling along Z axis
translation_matrix_along_Z = lambda scale : torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, scale],
    [0, 0, 0, 1]
], dtype=torch.float32)

# Rotation along X axis
rotation_matrix_along_X = lambda theta : torch.tensor([
    [1, 0, 0, 0],
    [0, np.cos(theta), -np.sin(theta), 0],
    [0, np.sin(theta), np.cos(theta), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

# Rotation along Y axis
rotation_matrix_along_Y = lambda theta : torch.tensor([
    [np.cos(theta), 0, -np.sin(theta), 0],
    [0, 1, 0, 0],
    [np.sin(theta), 0, np.cos(theta), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

def get_transformation_matrix(scale, phi, theta):
    matrix = translation_matrix_along_Z(scale) @ rotation_matrix_along_X(phi) @ rotation_matrix_along_Y(theta)
    # Changes to the co-ordinate axis: Invert X axis, Switch Y and Z axis
    change_matrix = torch.Tensor([[-1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 0, 1]])
    return matrix @ change_matrix

# Code to load blender dataset
class BlenderDataset(Dataset):
    def __init__(self, basedir, split, resolution):
        super(BlenderDataset, self).__init__()
        json_data = json.load(open(os.path.join(basedir, "transforms_"+split+".json"), 'r'))

        self.images = []
        self.poses = []
        for frame in json_data['frames']:
            self.images.append(os.path.join(basedir, frame["file_path"]+".png"))
            self.poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

        self.camera_angle_x = json_data["camera_angle_x"]
        # Compute focal length in pixels, focal_length = (resolution/2) / tan(camera_angle/2)
        self.focal_length = (0.5 * resolution) / np.tan(0.5 * self.camera_angle_x)

        # The image is rendered from point looking downward by -30 deg on X axis
        self.render_poses = torch.stack(
            tensors=[get_transformation_matrix(scale=4, phi=-30, theta=angle) for angle in np.linspace(start=-180, stop=180, num=41)[:-1]],
            dim=0                          
        )

        self.images_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images_transforms(Image.open(self.images[index]))
        pose = self.poses[index]

        height, width = image.shape[:2]

        return image[:3,...], pose, self.render_poses, [height, width, self.focal_length]
    
if __name__ == "__main__":
    dataset = BlenderDataset(basedir=r"D:\CV_Projects\NeRF-PyTorch\data\lego", split="train", resolution=400)
    print("Length of dataset", len(dataset))
    image, pose, render_poses, [height, width, focal_length] = dataset[0]
    print("Image resolution: ", image.shape)
    print("Pose: ", pose)
    print("Render Poses: ", render_poses.shape)
    print("Focal length: ", focal_length)