import os
import torch
from tqdm import tqdm
from torch.optim import optimizer

class NeRFTrainer:
    def __init__(
        self,
        coarse_model,
        fine_model,
        optimizer,
        loss_function,
        training_dataloader,
        num_epochs=100,
        save_frequency=25,
        device="cuda",
        model_dir="checkpoints",
        load_model=False,
        load_model_path="",
    ):
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        self.coarse_model = (
            coarse_model.to(self.device)
            if coarse_model is not None
            else self.raise_exception("Coarse Model is None")
        )
        self.fine_model = (
            fine_model.to(self.device)
            if fine_model is not None
            else self.raise_exception("Fine Model is None")
        )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else self.raise_exception("Optimizer is None")
        )
        self.loss_function = (
            loss_function.to(self.device)
            if loss_function is not None
            else self.raise_exception("Loss Function is None")
        )
        self.training_dataloader = (
            training_dataloader
            if training_dataloader is not None
            else self.raise_exception("Training Dataloader is None")
        )
        self.num_epochs = num_epochs
        self.save_frequency = save_frequency
        self.model_dir = model_dir
        if load_model == True:
            if os.path.exists(load_model_path):
                self.load_model(load_model_path)
            else:
                self.raise_exception("Load Model path is empty")

    def raise_exception(self, message):
        raise Exception(message)
    
    def render(self, rays):
        # Rays shape would be: [Batch_size, H*W, 2, 3]
        ray_origins, ray_directions = rays[:, :, 0, :], rays[:, :, 1, :]
        # Get view directions with shape: [Batch_size, H*W, 3]
        view_directions = ray_directions/torch.norm(ray_directions, dim=-1, keepdim=True)
        # Check if you need to reshape this above vector.

        # Reshaping them to [Batch_size * H*W, 3]
        ray_origins = torch.reshape(ray_origins, [-1, 3])
        ray_directions = torch.reshape(ray_directions, [-1, 3])
        view_directions = torch.reshape(view_directions, [-1, 3])

        # Near and Far vectors to get the Z values of the scene to query over.
        near_const = self.training_dataloader.dataset.near_const # 2.0 for blender
        far_const = self.training_dataloader.dataset.far_const # 6.0 for blender
        # Shape of the below vectors should be: [Batch_size*H*W, 1]
        near = near_const * torch.ones_like(ray_directions[..., :1])
        far = far_const * torch.ones_like(ray_directions[..., :1])

        # This is the part where we do the sampling of the rays which for different Z-values.
        # This is done coarsely and then finely to avoid empty space sampling.
        sample_size = 64
        interval_values = torch.linspace(0.0, 1.0, steps=sample_size)
        z_values = near * (1.0-interval_values) + far * (interval_values)

        # Based on the paper, we sample Z_values in the intervals randomly.
        # Since random values are between -1 to 1, we need to do this as follows,
        # l, u - lower and upper range of interval. m - mid value of the interval
        # l + (m * random_number) - randomly samples the number greater than 
        # lower range and adds or subtracts mid_value*random_number. +/- based on random number.
        mids = 0.5 * (z_values[..., 1:] + z_values[..., :-1])
        upper = torch.cat([mids, z_values[..., -1:]], dim=-1)
        lower = torch.cat([z_values[...,:1], mids], dim=-1)
        random_values = torch.randn(z_values.shape)

        z_values = lower + (upper - lower) * random_values

        # From the ray origin points, in the direction, we get the point by multiplying the 
        # Z values. Shapes change as r_d: [Batch_size*H*W, 3] -> [Batch_size*H*W, 1, 3]
        # Z_values: [Batch_size*H*W, 64] -> [Batch_size*H*W, 64 ,1]
        # Eventually points: [Batch_size*H*W, 64, 3]
        points = ray_origins[..., None, :] + ray_directions[..., None, :] * z_values[..., :, None]

        points_flattened = torch.reshape(points, [-1, 3]) # Shape: [Batch_size*H*W*64, 3]
        view_directions = view_directions[:, None, 3].expand(points.shape) # Shape: [Batch_size*H*W , 64, 3]
        view_directions_flattened = torch.reshape(view_directions, [-1, 3]) # Shape: [Batch_size*H*W*64, 3]

        coarse_network_output = self.coarse_model(points_flattened, view_directions_flattened)

        # Now that we have coarse network outputs we can sample finer points 
        # along Z axis and use the fine network to get better results.
        


        

    def train_one_epoch(self):
        average_loss = 0
        for data in self.training_dataloader:
            self.optimizer.zero_grad()

            # Do something with the data
            # Shape would be : [Batch_size, H*W, r_o + r_d + rgb, 3]
            image, pose, rays_with_target = data
            rays, target_rgb = rays_with_target[:, :, :-1], rays_with_target[:, :, -1]

            output = self.render(rays)

            # Pass appropriate elements in the loss function.
            loss = self.loss_function(...)
            loss.backward()
            self.optimizer.step()
            average_loss += loss.item()

        average_loss = average_loss / len(self.training_dataloader)

        return average_loss

    def train(self):
        progress_bar = tqdm(range(self.num_epochs))
        for epoch in progress_bar:
            train_loss = self.train_one_epoch()
            progress_bar.set_description(
                f"Epoch: {epoch+1}/{self.num_epochs} Training Loss: {train_loss} "
            )
            if (epoch + 1) % self.save_frequency == 0:
                self.save_model(epoch + 1)

    def save_model(self, epoch):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            {
                "epoch": epoch,
                "coarse_model_state_dict": self.coarse_model.state_dict(),
                "fine_model_state_dict": self.fine_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            os.path.join(self.model_dir, f"checkpoint_{epoch}.pt"),
        )

    def load_model(self, model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        # TODO: You might want to use the epoch stored in it too.
        self.coarse_model.load_state_dict(checkpoint["coarse_model_state_dict"])
        self.fine_model.load_state_dict(checkpoint["fine_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])