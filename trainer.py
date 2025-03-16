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
        initial_learning_rate=5e-4,
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
        self.initial_learning_rate = initial_learning_rate
        self.num_epochs = num_epochs
        self.save_frequency = save_frequency
        self.model_dir = model_dir
        if load_model == True:
            self.load_model(load_model_path)

    def raise_exception(self, message):
        raise Exception(message)

    def run_network_in_batches(
        self, points, view_directions, network, ray_batch_size=1024
    ):
        batch_indices = [
            i * ray_batch_size for i in range(points.shape[0] // ray_batch_size)
        ] + [points.shape[0]]
        outputs = [
            (
                self.coarse_model(points[start:end], view_directions[start:end])
                if network == "coarse"
                else self.fine_model(points[start:end], view_directions[start:end])
            )
            for start, end in zip(batch_indices, batch_indices[1:])
        ]
        outputs = torch.cat(tensors=outputs, dim=0)
        return outputs

    def render(self, rays, coarse_sample_size=64, fine_sample_size=128):
        batch_size, pixel_count, _, _ = rays.shape
        # Rays shape would be: [Batch_size, H*W, 2, 3]
        ray_origins, ray_directions = rays[:, :, 0, :], rays[:, :, 1, :]
        # Get view directions with shape: [Batch_size, H*W, 3]
        view_directions = ray_directions / torch.norm(
            ray_directions, dim=-1, keepdim=True
        )
        # Check if you need to reshape this above vector.

        # Reshaping them to [Batch_size * H*W, 3]
        ray_origins = torch.reshape(ray_origins, [-1, 3])
        ray_directions = torch.reshape(ray_directions, [-1, 3])
        view_directions = torch.reshape(view_directions, [-1, 3])

        # Near and Far vectors to get the Z values of the scene to query over.
        near_const = self.training_dataloader.dataset.near_const  # 2.0 for blender
        far_const = self.training_dataloader.dataset.far_const  # 6.0 for blender
        # Shape of the below vectors should be: [Batch_size*H*W, 1]
        near = near_const * torch.ones_like(ray_directions[..., :1]).to(self.device)
        far = far_const * torch.ones_like(ray_directions[..., :1]).to(self.device)

        # This is the part where we do the sampling of the rays which for different Z-values.
        # This is done coarsely and then finely to avoid empty space sampling.
        interval_values = torch.linspace(0.0, 1.0, steps=coarse_sample_size).to(
            self.device
        )
        z_values = near * (1.0 - interval_values) + far * (
            interval_values
        )  # Shape: [Batch_size*H*W, 64]

        # Based on the paper, we sample Z_values in the intervals randomly.
        # Since random values are between -1 to 1, we need to do this as follows,
        # l, u - lower and upper range of interval. m - mid value of the interval
        # l + (m * random_number) - randomly samples the number greater than
        # lower range and adds or subtracts mid_value*random_number. +/- based on random number.
        mids = 0.5 * (z_values[..., 1:] + z_values[..., :-1])
        upper = torch.cat([mids, z_values[..., -1:]], dim=-1)
        lower = torch.cat([z_values[..., :1], mids], dim=-1)
        random_values = torch.randn(z_values.shape).to(self.device)

        # This is coarsely sampled Z values.
        z_values = lower + (upper - lower) * random_values

        # From the ray origin points, in the direction, we get the point
        # by multiplying the Z values. Shapes change for concat as
        # ray_directions: [Batch_size*H*W, 3] -> [Batch_size*H*W, 1, 3]
        # Z_values: [Batch_size*H*W, 64] -> [Batch_size*H*W, 64 ,1]
        # Eventually points: [Batch_size*H*W, 64, 3]
        points = (
            ray_origins[..., None, :]
            + ray_directions[..., None, :] * z_values[..., :, None]
        )

        points_flattened = torch.reshape(
            points, [-1, 3]
        )  # Shape: [Batch_size*H*W*64, 3]
        view_directions = view_directions[:, None, :].expand(
            points.shape
        )  # Shape: [Batch_size*H*W , 64, 3]
        view_directions_flattened = torch.reshape(
            view_directions, [-1, 3]
        )  # Shape: [Batch_size*H*W*64, 3]

        # Calling the Coarse Network here.
        coarse_network_output = self.run_network_in_batches(
            points=points_flattened, 
            view_directions=view_directions_flattened,
            network="coarse"
        )
        coarse_network_output_colors = torch.reshape(
            coarse_network_output[0], points.shape
        )
        coarse_network_output_density = torch.reshape(
            coarse_network_output[1],
            list(points.shape[:-1]) + [coarse_network_output[1].shape[-1]],
        )

        # Now that we have coarse network outputs we can sample finer points
        # along Z axis and use the fine network to get better results.
        coarse_color_map, disparity_map, accumulation_map, weights, depth_map = (
            self.transform_network_outputs(
                colors=coarse_network_output_colors,
                density=coarse_network_output_density,
                z_values=z_values,
                ray_directions=ray_directions,
            )
        )

        # Now that we have the outputs from coarse network, we can proceed to
        # the finer model with a higher sample size based on the coarse model output.
        # Since Z values mids is of the shape [Batch_size*H*W, 63], we ignore the weights value at 0th index
        finer_z_values = self.sample_finer_z_values(
            bins=mids, weights=weights[..., 1:], fine_sample_size=fine_sample_size
        )

        # TODO: Not sure why I am doing the detach below. Fix or explain why later.
        finer_z_values = finer_z_values.detach()
        z_values, indices = torch.sort(
            input=torch.cat(tensors=[z_values, finer_z_values], dim=-1),
            dim=-1,  # Sort based on the Z_values to interleave the
            # finer Z values in between the coarse ones.
        )

        # Get points with finer Z values.
        points = (
            ray_origins[..., None, :]
            + ray_directions[..., None, :] * z_values[..., :, None]
        )

        points_flattened = torch.reshape(
            points, [-1, 3]
        )  # Shape: [Batch_size*H*W*(64+128), 3]
        view_directions = view_directions[:, None, 3].expand(
            points.shape
        )  # Shape: [Batch_size*H*W , (64+128), 3]
        view_directions_flattened = torch.reshape(
            view_directions, [-1, 3]
        )  # Shape: [Batch_size*H*W*(64+128), 3]

        fine_network_output = self.run_network_in_batches(
            points=points_flattened, 
            view_directions=view_directions_flattened,
            network="fine"
        )
        fine_network_output_colors = torch.reshape(fine_network_output[0], points.shape)
        fine_network_output_density = torch.reshape(
            fine_network_output[1],
            list(points.shape[:-1]) + [fine_network_output[1].shape[-1]],
        )

        fine_color_map, disparity_map, accumulation_map, weights, depth_map = (
            self.transform_network_outputs(
                colors=fine_network_output_colors,
                density=fine_network_output_density,
                z_values=z_values,
                ray_directions=ray_directions,
            )
        )

        render_outputs = {
            "coarse_colors": torch.reshape(
                coarse_color_map, [batch_size, pixel_count, 3]
            ),
            "fine_colors": torch.reshape(fine_color_map, [batch_size, pixel_count, 3]),
            "depths": torch.reshape(depth_map, [batch_size, pixel_count, 1]),
            "disparities": torch.reshape(disparity_map, [batch_size, pixel_count, 1]),
            "accumulations": torch.reshape(
                accumulation_map, [batch_size, pixel_count, 1]
            ),
        }

        return render_outputs

    def sample_finer_z_values(self, bins, weights, fine_sample_size):
        # We add a small value to prevent NaNs.
        weights = weights + 1e-5  # Shape: [Batch_size*H*W, 63]
        probability_distribution = weights / torch.sum(
            input=weights, dim=-1, keepdim=True
        )
        cumulative_distribution = torch.cumsum(input=probability_distribution, dim=-1)
        # Set the first value of the cumulative distribution as zero.
        cumulative_distribution = torch.cat(
            tensors=[
                torch.zeros_like(
                    cumulative_distribution[..., :0]
                ),  # Shape : [Batch_size*H*W, 1]
                cumulative_distribution,
            ],
            dim=-1,
        )  # Shape: [Batch_size*H*W, 64]

        # Sample finer random values. Shape: [Batch_size*H*W, 128]
        samples = torch.randn(
            list(cumulative_distribution.shape[:-1]) + [fine_sample_size]
        ).to(self.device)

        # Get the right most index where the samples fit in the cumulative distribuition.
        indices = torch.searchsorted(
            sorted_sequence=cumulative_distribution, input=samples, right=True
        )  # Shape: [Batch_size*H*W, 128]
        # Get one index lesser than the search sorted index. Use zero if index goes negative.
        below_indices = torch.max(torch.zeros_like(input=indices), indices - 1)
        # Now cap the search sorted index so that it doesn't exceed the highest index.
        # Use last index(last index shape-1) if index goes too high.
        above_indices = torch.min(
            (cumulative_distribution.shape[-1] - 1) * torch.ones_like(input=indices),
            indices,
        )

        # To use the indices we obtained previously we need to expand the
        # cumulative distribution tensor and the bins tensor to the shapes of the indices.
        # And then obtain the values using torch.gather which does the work for us.
        shape_to_expand_by = [
            indices.shape[0],
            indices.shape[1],
            cumulative_distribution.shape[-1],
        ]
        below_cumulative_distribution_values = torch.gather(
            input=cumulative_distribution[:, None, ...].expand(shape_to_expand_by),
            dim=2,
            index=below_indices[..., None],
        )  # Shape: [Batch_size*H*W, 128, 1]
        above_cumulative_distribution_values = torch.gather(
            input=cumulative_distribution[:, None, ...].expand(shape_to_expand_by),
            dim=2,
            index=above_indices[..., None],
        )  # Shape: [Batch_size*H*W, 128, 1]

        # Similarly for bins
        below_bin_values = torch.gather(
            input=bins[:, None, ...].expand(shape_to_expand_by),
            dim=2,
            index=below_indices[..., None],
        )  # Shape: [Batch_size*H*W, 128, 1]
        above_bin_values = torch.gather(
            input=bins[:, None, ...].expand(shape_to_expand_by),
            dim=2,
            index=above_indices[..., None],
        )  # Shape: [Batch_size*H*W, 128, 1]

        # Get the interval size between the cumulative distribution values to accomodate the samples.
        denominator = (
            above_cumulative_distribution_values[..., 0]
            - below_cumulative_distribution_values[..., 0]
        )
        # Make sure there are no zeros, so replace them with ones.
        denominator = torch.where(
            condition=denominator < 1e-5,
            input=torch.ones_like(
                denominator
            ),  # If condition is met get values from input tensor.
            other=denominator,  # If condition is not met get values from other tensor.
        )

        # Now that we know the interval size(i.e. denominator) and
        # lower end value of the interval which is the below_cumulative_distribution_values.
        # We find where each sample is going to lie in its interval.
        interval_position_of_sample = (
            samples - below_cumulative_distribution_values[..., 0]
        ) / denominator

        # Now to the actual Z values by using the bins and multiply it with the interval position of sample.
        finer_z_values = below_bin_values[..., 0] + interval_position_of_sample * (
            above_bin_values[..., 0] - below_bin_values[..., 0]
        )

        return finer_z_values  # Shape: [Batch_size*H*W, 128]

    def transform_network_outputs(self, colors, density, z_values, ray_directions):
        # Distances: it is the distance between adjacent samples. This is
        # from the volume rendering equation represented by delta(i).
        distances = z_values[..., 1:] - z_values[..., :-1]
        # Since this is one less than the sample size, we concat one huge value at the end.
        infinity = (
            torch.Tensor([1e10])
            .expand(list(distances.shape[:-1]) + [1])
            .to(self.device)
        )
        distances = torch.cat([distances, infinity], dim=-1)
        # Find the norm of the ray directions individually and multiply with the
        # values obtained from Z_values.
        distances = distances * torch.norm(ray_directions[..., None, :], dim=-1)

        # Colors have been already passed through Sigmoid layer in the network.
        # Density Values are also obtained after ReLU activation.
        # We compute alpha values based on the formula from paper.
        # Density Shape: [Batch_size*H*W, 64, 1]
        alpha = 1.0 - torch.exp(
            -1 * density[..., 0] * distances
        )  # Shape: [Batch_size*H*W, 64].

        # This is computed in a trivial fashion, as opposed to the formula
        # given in the paper. But when simplified it is the same.
        # Note: This transmittance is along the ray(near to far) and not among multiple rays.
        # transmittance = exp(-1 * cummulative SUM of the product of density and distances).
        # which is also the same as cummulative PRODUCT of the exponent of (-1*density*distance).
        # We already computed the product of density and distance for alpha in the previous step.
        # We can make use of this by subtracting 1 to it and get just the product value.
        # (1 - alpha) i.e. 1 - (1.0 - exp(-1*density*distances)) = exp(-1*density*distances)
        # Transmittance initial value i.e. T(i) when i=0 will be 1. so we concat
        # torch.ones of shape [Batch_size*H*W, 1] with [Batch_size*H*W, 64] to get
        # [Batch_size*H*W, 65] and then get rid of the last value to retain only 64 samples.
        accumulated_transmittance = torch.cumprod(
            torch.cat(
                tensors=[
                    torch.ones((alpha.shape[0], 1)).to(self.device),
                    1.0 - alpha + 1e-10,  # so that Nan values are avoided.
                ],
                dim=-1,
            ),  # Concat on the last dimension to get shape: [Batch_size*H*W, 65]
            dim=-1,  # Cummulative product on the last dimension.
        )
        accumulated_transmittance = accumulated_transmittance[
            ..., :-1
        ]  # Shape: [Batch_size*H*W, 64]

        weights = alpha * accumulated_transmittance
        # Summing over the ray samples to get [Batch_size*H*W, 3]
        # [Batch_size*H*W, 64, 1] * [Batch_size*H*W, 64, 3] and sum on dim with size 64 reduced to 1. But keep in mind the final shape is [Batch_size*H*W, 3].
        color_map = torch.sum(input=weights[..., None] * colors, dim=-2)
        depth_map = torch.sum(input=weights * z_values, dim=-1)

        # Accumulation Map: This is the sum of the weights along each ray, giving an idea
        # of how much of the total ray is occupied by the scene versus being empty.
        # Basically, How much of the scene is covered by the ray (opacity measure).
        accumulation_map = torch.sum(input=weights, dim=-1)

        # Disparity Map: The disparity map is the inverse of the depth map.
        # It provides a representation of how close objects are to the camera.
        # Basically, How far objects are (inverse depth).
        disparity_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / accumulation_map
        )

        # Code to make the background white.
        color_map = color_map + (1.0 - accumulation_map[..., None])

        return color_map, disparity_map, accumulation_map, weights, depth_map

    def train_one_epoch(self):
        average_loss = 0
        for data in tqdm(self.training_dataloader):
            self.optimizer.zero_grad()

            # Do something with the data
            # Shape would be : [Batch_size, H*W, r_o + r_d + rgb, 3]
            image, pose, rays_with_target = data
            rays_with_target = rays_with_target.to(self.device)
            rays, target_colors = (
                rays_with_target[:, :, :-1],
                rays_with_target[:, :, -1],
            )

            output = self.render(rays)

            # Pass appropriate elements in the loss function.
            coarse_model_loss = self.loss_function(
                output["coarse_colors"], target_colors
            )
            fine_model_loss = self.loss_function(output["fine_colors"], target_colors)
            loss = coarse_model_loss + fine_model_loss
            loss.backward()
            self.optimizer.step()
            average_loss += loss.item()

            # TODO: Include a learning rate scheduler here. VERY IMPORTANT !!!

        average_loss = average_loss / len(self.training_dataloader)

        return average_loss

    def train(self):
        progress_bar = tqdm(range(self.num_epochs))
        for epoch in progress_bar:
            train_loss = self.train_one_epoch()
            # Using the same kind of learning rate scheduling as the Authors.
            decay_rate = 0.1
            decay_steps = 250 * 1000
            new_learning_rate = self.initial_learning_rate * (
                decay_rate ** ((epoch + 1) / decay_steps)
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_learning_rate
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
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.model_dir, f"checkpoint_{epoch}.pt"),
        )

    def load_model(self, model_path):
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path)
            # TODO: You might want to use the epoch stored in it too.
            self.coarse_model.load_state_dict(checkpoint["coarse_model_state_dict"])
            self.fine_model.load_state_dict(checkpoint["fine_model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            self.raise_exception("Load Model path is empty")
