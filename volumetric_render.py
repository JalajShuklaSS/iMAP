import torch


def volume_render(distances, sigmas):
        """
        Perform volume rendering to generate depth and intensity maps.

        Parameters:
        distances: A tensor containing distances to sampled points along rays.
        sigmas: A tensor containing density values at the sampled points.

        Returns:
        depth_map: The rendered depth map, representing distances along rays.
        intensity_map: The rendered intensity map, representing brightness.
        depth_variance: The variance of the rendered depth map.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # here we assume that the maximum distance is 1.5
        max_distance = 1.5
        batch_size = distances.shape[0]
        # Compute the number of steps along each ray
        num_steps = distances.shape[1]

        # Here we are getting the step sizes along the rays
        step_sizes = distances[:, 1:] - distances[:, :-1]

        # We want to calculate the opacity of each step
        # Initially compute the density values multiplied by step sizes
        density_step_product = sigmas[:, :-1, 3] * step_sizes

        # Here we are getting the exponent term
        exponent_term = -density_step_product

        # Here we need to compute the negative exponential
        negative_exponential = torch.exp(exponent_term)

        # Finally we are getting the opacity
        opacity = 1 - negative_exponential

        # Get the accumulated opacity along each ray
        accumulated_opacity = torch.cumprod(1 - opacity, dim=1)

        # We are calculating the weights for each step
        weights = torch.zeros((batch_size, num_steps - 1), device=device)
        weights[:, 1:] = opacity[:, 1:] * accumulated_opacity[:, :-1]

        # Here we are getting the depth and intensity maps
        depth_map = torch.sum(weights * distances[:, :-1], dim=1)
        intensity_map = torch.sum(weights.view(batch_size, -1, 1) * sigmas[:, :-1, :3], dim=1)

        # Here we are calcualting the variance of the depth map
        squared_diff = torch.square(distances[:, :-1] - depth_map.view(batch_size, 1))
        # Compute the depth variance
        depth_variance = torch.sum(weights * squared_diff, dim=1)
        # Here we need to add a background depth and intensity for pixels that don't hit any surface
        depth_map += accumulated_opacity[:, -1] * max_distance

        return depth_map, intensity_map, depth_variance