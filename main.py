import torch
import numpy as np
import torch.optim as optim
from model import NeRF_model
from dataloader import BlenderDataset
from trainer import NeRFTrainer
from torch.utils.data import DataLoader

#
basedir = r"D:\CV_Projects\NeRF-PyTorch\data\lego"

# Define Datasets
train_dataset, train_render_poses, train_data_intrinsics = (
    BlenderDataset.create_and_return_object(
        basedir=basedir, split="train", resolution=400
    )
)
val_dataset, val_render_poses, val_data_intrinsics = (
    BlenderDataset.create_and_return_object(
        basedir=basedir, split="val", resolution=400
    )
)
test_dataset, test_render_poses, test_data_intrinsics = (
    BlenderDataset.create_and_return_object(
        basedir=basedir, split="test", resolution=400
    )
)

height, width, focal_length = train_data_intrinsics
camera_intrinsic_matrix = np.array(
    [[focal_length, 0, 0.5 * width], [0, focal_length, 0.5 * height], [0, 0, 1]]
)

# Define Dataloaders
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create Nerf Model
coarse_nerf_model = NeRF_model()
fine_nerf_model = NeRF_model()
trainable_params = list(coarse_nerf_model.parameters()) + list(
    fine_nerf_model.parameters()
)

#
learning_rate = 5e-4

# Define Optimizer
optimizer = optim.Adam(params=trainable_params, lr=learning_rate, betas=(0.9, 0.999))

# Loss function
loss_function = torch.nn.MSELoss()

# Trainer
nerf_trainer = NeRFTrainer(
    coarse_model=coarse_nerf_model,
    fine_model=fine_nerf_model,
    optimizer=optimizer,
    loss_function=loss_function,
    training_dataloader=train_dataloader,
)

nerf_trainer.train()
