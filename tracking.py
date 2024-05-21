import torch 
from mapper import Mapper

def freeze_model_for_tracking(self):
    self.model_tracking.load_state_dict(copy.deepcopy(self.model.state_dict()))

def track(self, camera, batch_size=200, n_iters=20):
    """

    Track the movement of a camera view to align it with the ground truth depth values. This function takes camera parameters and the batch size of rays as input.

    Parameters:

    camera: Camera parameters representing the view to be tracked.
    batch_size: The number of rays in a batch used for tracking.
    Returns:

    p: The proportion of depth values that closely match the ground truth depth values. This value indicates the accuracy of the tracking process.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Update the model for tracking
    self.freeze_model_for_tracking()
    depth_gt_list = []
    rgb_gt_list = []
    for _ in range(n_iters):
        # Reset the gradients
        camera.optimizer.zero_grad()

        # Update the camera transform
        camera.update_transform()

        # Compute the size of the image
        h, w = camera.image_size

        # Compute the pixel coordinates for a randomly sampled batch of rays
        u_coordinates = (torch.rand(batch_size) * (w - 1)).to(torch.int16).to(device)
        v_coordinates = (torch.rand(batch_size) * (h - 1)).to(torch.int16).to(device)

        # Render the batch of rays
        depth, rgb, depth_variance = self.render_rays(u_coordinates, v_coordinates, camera, use_tracking_model=True)

        # Get the ground truth depth and RGB values
        for u, v in zip(u_coordinates, v_coordinates):
            depth_gt_list.append(camera.depth_image[v, u].unsqueeze(0))
            rgb_gt_list.append(camera.rgb_image[v, u, :].unsqueeze(0))

        depth_gt = torch.cat(depth_gt_list)
        rgb_gt = torch.cat(rgb_gt_list)
        # Ignore the depth values for pixels that don't hit any surface
        depth[depth_gt == 0] = 0

        # Compute the inverse variance
        with torch.no_grad():
            sqrt_depth_variance = torch.sqrt(depth_variance)
            inverse_variance = torch.reciprocal(sqrt_depth_variance)
            inverse_variance[inverse_variance.isinf()] = 1
            inverse_variance[inverse_variance.isnan()] = 1

        # The depth loss is inversely weighted by the depth variance
        depth_loss = torch.mean(torch.abs(depth - depth_gt) * inverse_variance)

        # The RGB loss is weighted as more important by a factor of 5
        rgb_loss = 5 * torch.mean(torch.abs(rgb - rgb_gt))

        total_loss = depth_loss + rgb_loss

        # Backpropagate the loss and update the parameters
        total_loss.backward()
        camera.optimizer.step()

    # Compute the proportion of depth values that are close to the ground truth
        # Calculate Absolute Difference
        abs_diff = torch.abs(depth - depth_gt)
        # Calculate Reciprocal
        reciprocal_depth_gt = torch.reciprocal(depth_gt)
        # Add Small Constant
        reciprocal_depth_gt = reciprocal_depth_gt + 1e-12
        # Element-wise Multiplication
        mul_result = abs_diff * reciprocal_depth_gt
        # Check Threshold
        threshold_result = mul_result < 0.1
        # Convert to Integer
        int_result = threshold_result.int()
        # Summation
        sum_result = torch.sum(int_result)
        # CPU Conversion
        cpu_result = sum_result.cpu()
        # Convert to Float
        float_result = float(cpu_result.item())
        # Calculate Proportion
        p = float_result / batch_size
        # If the proportion is high enough, short circuit stop the tracking
        if p > 0.8:
            break

    print("It is working and Tracking: P=", p)

    return p