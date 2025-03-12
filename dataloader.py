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
        # Near and Far values specific to dataset.
        self.near_const = 2.0
        self.far_const = 6.0
        self.resolution = resolution
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

        self.camera_intrinsic_matrix = np.array([
            [self.focal_length, 0, 0.5*self.resolution],
            [0, self.focal_length, 0.5*self.resolution],
            [0, 0, 1]
        ])

        self.prepare_utils()

    def prepare_utils(self):
        # i, j used for camera co-ordinates
        camera_coords_i, camera_coords_j = torch.meshgrid(
            torch.linspace(start=0, end=self.resolution-1, steps=self.resolution),
            torch.linspace(start=0, end=self.resolution-1, steps=self.resolution)
        )

        # Normalize to camera space: 
        # 1. centre the image(convert pixel coords(0,0)...(H,W) to a system with (0,0) at the image centre) 
        # 2. normalize to camera space by dividing with focal length,
        # 3. Set Z = -1 the direction in which the camera is looking at the scene
        self.directions = torch.stack(
            [
                (camera_coords_i - 0.5*self.resolution)/self.focal_length, # X
                -(camera_coords_j - 0.5*self.resolution)/self.focal_length, # Y
                -torch.ones_like(camera_coords_i) # Z
            ],
            dim = -1 # Stack on the last dimension
        )


    @classmethod
    def create_and_return_object(classname, basedir, split, resolution):
        dataset_object = classname(basedir, split, resolution)
        return dataset_object, dataset_object.render_poses, [dataset_object.resolution, dataset_object.resolution, dataset_object.focal_length]

    def __len__(self):
        return len(self.images)
    
    def get_rays(self, pose):
        # Now we convert each ray direction vector which is in the camera space to the world 
        # space by using the pose matrix (3, 3) which is a transformation matrix converting
        # the rays into the world space.
        ray_directions = torch.sum(
            self.directions[..., None, :] * pose[:3, :3],
            dim=-1
        )

        # Rays origin is the same for all the rays, so just make 
        # copies of it to match the ray_directions shape.
        ray_origins = torch.tensor(pose[:3, -1]).expand(ray_directions.shape)

        return ray_origins, ray_directions

    def __getitem__(self, index):
        image = self.images_transforms(Image.open(self.images[index]))[:3,...]
        pose = self.poses[index]

        ray_origins, ray_directions = self.get_rays(pose)

        # Concat the ray origin, ray direction and the rgb from image. 
        # Shape: [r_o + r_d + rgb, H, W, 3]
        rays = torch.cat(
            tensors=[
                ray_origins[None, ...], 
                ray_directions[None, ...],
                torch.reshape(image, (-1, self.resolution, self.resolution, 3))
            ],
            dim = 0
        )
        # ReShape to : [H, W, r_o + r_d + rgb, 3]
        rays = torch.permute(rays, (1, 2, 0, 3))
        # ReShape to : [H * W, (r_o + r_d + rgb), 3]
        rays = torch.reshape(rays, (-1, 3, 3))

        # Returning the image, the pose w.r.t image and the rays specific to the image
        return image, pose, rays
    
if __name__ == "__main__":
    dataset, render_poses, [height, width, focal_length] = BlenderDataset.create_and_return_object(basedir=r"D:\CV_Projects\NeRF-PyTorch\data\lego", split="train", resolution=400)
    print("Length of dataset", len(dataset))
    image, pose, rays = dataset[0]
    print("Image resolution: ", image.shape)
    print("Pose: ", pose)
    print("Rays: ", rays.shape)
    print("Render Poses: ", render_poses.shape)
    print("Focal length: ", focal_length)