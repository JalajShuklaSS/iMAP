import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import time
import os
from weights_sample import Weightes_sample
from model import Camera, IMAP
from volumetric_render import volume_render

class Mapper():
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = IMAP().to(device)
        self.model_tracking = IMAP().to(device)
        self.cameras = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.render_id=0

    def freeze_model_for_tracking(self):
        self.model_tracking.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def add_camera(self, rgb_filename, depth_filename, position_x, position_y, position_z, rotation_x, rotation_y, rotation_z, light_scale, light_offset):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rgb = cv2.imread(rgb_filename, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
        camera = Camera(rgb, depth, 
                        position_x, position_y, position_z, 
                        rotation_x, rotation_y, rotation_z, 
                        light_scale, light_offset)
        self.cameras.append(camera)




    def render_rays(self, pixel_u, pixel_v, cam, coarse_samples=32, fine_samples=12, tracking_model=False):
        """
    Render a batch of rays to generate depth and intensity maps.

        Parameters:
        pixel_u: The horizontal pixel coordinates of the rays.
        pixel_v: The vertical pixel coordinates of the rays.
        cam: An object containing camera parameters, such as position and orientation.
        coarse_samples: The number of coarse samples to be taken along each ray.
        fine_samples: The number of fine samples to be taken along each ray.
        tracking_model: A boolean flag indicating whether to use the tracking model.

        Returns:
        depth_map: The rendered depth map, representing the distance of objects from the camera.
        intensity_map: The rendered intensity map, representing the brightness of objects in the scene.
        depth_variance: The variance of the rendered depth map, providing information about the uncertainty in depth estimation.
      """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if tracking_model:
            model = self.model_tracking
        else:
            model = self.model

        batch_size = pixel_u.shape[0]

        # get the rays' direction and origin
        ray_direction, ray_origin = cam.rays_for_pixels(pixel_u, pixel_v)

        # we first perform coarse sampling to determine the fine sample locations so as to save computation
        with torch.no_grad(): 
        # Part 1: Generate a 1D tensor of equally spaced values from 0.0001 to 1.2
            distances_coarse_1d = torch.linspace(0.0001, 1.2, coarse_samples, device=device)

        # Part 2: Reshape the 1D tensor into a 2D tensor with shape (1, coarse_samples)
            distances_coarse_2d = distances_coarse_1d.view(1, coarse_samples)

        # Part 3: Expand the 2D tensor along the batch dimension to match the batch size
            distances_coarse = distances_coarse_2d.expand(batch_size, coarse_samples)
            
        # Reshape ray_origin to have shape (batch_size, 1, 3)
            ray_origin_reshaped = ray_origin.view(batch_size, 1, 3)

        # Reshape ray_direction to have shape (batch_size, 1, 3)
            ray_direction_reshaped = ray_direction.view(batch_size, 1, 3)

        # Reshape distances_coarse to have shape (batch_size, coarse_samples, 1)
            distances_coarse_reshaped = distances_coarse.view(batch_size, coarse_samples, 1)

        # Element-wise multiplication of ray_direction_reshaped and distances_coarse_reshaped
            scaled_direction = ray_direction_reshaped * distances_coarse_reshaped

         # Add scaled_direction to ray_origin_reshaped
            rays_coarse = ray_origin_reshaped + scaled_direction
        
        # we get the basic density values for the coarse samples
            sigmas_coarse = model(rays_coarse.view(-1, 3)).view(batch_size, coarse_samples, 4)

        # compute the weights for the hierarchical sampling
            step_size = distances_coarse[0, 1] - distances_coarse[0, 0]
        # Compute opacity using sigmas_coarse
            opacity_part1 = -sigmas_coarse[:, :, 3] * step_size
            opacity_part2 = torch.exp(opacity_part1)
            opacity = 1 - opacity_part2[:, 1:]

        # Compute accumulated transparency using sigmas_coarse
            transparency_part1 = -sigmas_coarse[:, :, 3] * step_size
            transparency_part2 = torch.cumsum(transparency_part1, dim=1)
            transparency_part3 = torch.exp(transparency_part2)
            accumulated_transparency = 1 - transparency_part3[:, :-1]

        # we are performing the weighted sampling
            distances_fine = Weightes_sample.weighted_samples(distances_coarse, opacity * accumulated_transparency, fine_samples)

        # Reshape ray_direction and distances_fine for element-wise multiplication
            ray_direction_reshaped = ray_direction.view(batch_size, 1, 3)
            distances_fine_reshaped = distances_fine.view(batch_size, coarse_samples + fine_samples, 1)

        # Calculate rays_fine
            ray_direction_scaled = ray_direction_reshaped * distances_fine_reshaped
            rays_fine = ray_origin.view(batch_size, 1, 3) + ray_direction_scaled

        # Feed rays_fine into the model to get sigmas_fine
            rays_fine_flattened = rays_fine.view(-1, 3)
            sigmas_fine_flattened = model(rays_fine_flattened).view(batch_size, coarse_samples + fine_samples, 4)

        # volume rendering with the fine samples
        depth, intensity, depth_variance = volume_render(distances_fine, sigmas_fine_flattened)

        # Here the adjustment intensity based on the camera parameters
        intensity = cam.exp_a * intensity + cam.params[7]

        return depth, intensity, depth_variance


    def render_preview_image(self, camera_instance, image_label, scaling_factor=5):
        """
        Generate a preview image from a camera's view.

        Parameters:
        camera_instance: An object containing parameters of the camera, such as position and orientation.
        image_label: A label to identify the preview image.
        scaling_factor: A factor to scale down the preview image for better visualization.

        Returns:
        filename: The name of the generated preview image file, including its path.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            # Update the camera transformation
            camera_instance.update_transform()

            # Compute the size of the preview image
            full_h, full_w = camera_instance.image_size
            
            # Here we are gettint the pixel coordinates for the preview image
            # Compute the indices for vertical lines
            vertical_indices_raw = torch.arange(int(full_h / scaling_factor))
            vertical_indices_expanded = vertical_indices_raw.view(-1, 1).expand(-1, int(full_w / scaling_factor))
            vertical_indices_flattened = vertical_indices_expanded.reshape(-1)
            vertical_indices = (scaling_factor * vertical_indices_flattened).to(device)

            # Compute the indices for horizontal lines
            horizontal_indices_raw = torch.arange(int(full_w / scaling_factor))
            horizontal_indices_expanded = horizontal_indices_raw.view(1, -1).expand(int(full_h / scaling_factor), -1)
            horizontal_indices_flattened = horizontal_indices_expanded.reshape(-1)
            horizontal_indices = (scaling_factor * horizontal_indices_flattened).to(device)

            # Render the preview image
            depth_map, rgb_image, _ = self.render_rays(horizontal_indices, vertical_indices, camera_instance)
            depth_map = depth_map.view(-1, (int(full_w / scaling_factor)))
            rgb_image = rgb_image.view(-1, (int(full_w / scaling_factor)), 3)

            # Convert the depth map and the RGB image to numpy arrays
            # Clamp the RGB values between 0 and 255
            clamped_rgb_image = torch.clamp(rgb_image * 255, 0, 255)

            # Convert the clamped RGB tensor to a numpy array of type uint8
            rgb_image= clamped_rgb_image.detach().cpu().numpy().astype(np.uint8)

            # Clamp the depth map values between 0 and 255
            clamped_depth_map = torch.clamp(depth_map * 50000 / 256, 0, 255)

            # Convert the clamped depth map tensor to a numpy array of type uint8
            depth_map = clamped_depth_map.detach().cpu().numpy().astype(np.uint8)

            # Here we are getting the ground truth images
            # Clamp and convert ground truth RGB image
            rgb_image_gt = torch.clamp(camera_instance.rgb_image * 255, 0, 255)
            rgb_image_gt = rgb_image_gt.detach().cpu().numpy().astype(np.uint8)

            # Clamp and convert ground truth depth map
            depth_map_gt = torch.clamp(camera_instance.depth_image * 50000 / 256, 0, 255)
            depth_map_gt = depth_map_gt.detach().cpu().numpy().astype(np.uint8)

            # Concatenate rendered and ground truth images for comparison
            rgb_image_preview = cv2.hconcat([cv2.resize(rgb_image, (full_w, full_h)), rgb_image_gt])
            # Convert the depth map to a 3-channel image for concatenation
            # Resize the depth map to full width and height
            resized_depth_map = cv2.resize(depth_map, (full_w, full_h))

            # Concatenate the resized depth map with the ground truth depth map
            concatenated_depth_maps = cv2.hconcat([resized_depth_map, depth_map_gt])

            # Convert the concatenated depth maps to RGB format
            depth_map_preview = cv2.cvtColor(concatenated_depth_maps, cv2.COLOR_GRAY2RGB)  # Here we are concatenating the RGB image and the depth map
            preview_image = cv2.vconcat([rgb_image_preview, depth_map_preview])

            # Here we are saving the preview image
            cv2.imwrite("C:/Users/LEGION/Desktop/New folder/rgbd_dataset_freiburg1_teddy/pictures/{}_{:04}.png".format(image_label, self.render_id), preview_image)
            self.render_id += 1
            filepath = "C:/Users/LEGION/Desktop/New folder/rgbd_dataset_freiburg1_teddy/pictures/{}_{:04}.png".format(image_label,self.render_id)
            filename = os.path.basename(filepath)
            self.render_id+=1
            return  filename
            


    def update_map(self, batch_size=200, active_sampling=True):
        """
        Perform mapping using a batch of rays. This function is designed to map a batch of rays onto a scene. It takes two parameters:

        batch_size: Specifies the size of the batch of rays to be processed.
        active_sampling: Determines whether active sampling should be used during the mapping process.
        No value is returned by this function; instead, it modifies internal states and attributes to update the mapping based on the provided parameters.
        """
        # Here this is used in the device to be used for computations as we are working on google collab at times
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Select a set of images/cameras to use for the mapping
        if len(self.cameras) < 5:
            c_ids = np.arange(len(self.cameras))
        else:
            # Here we are selecting 5 different cameras on random, but we are making sure that the most recent two images are included
            c_ids = np.random.randint(0, len(self.cameras) - 2, 5)
            c_ids[3] = len(self.cameras) - 1
            c_ids[4] = len(self.cameras) - 2

        # Here we are performing the mapping for each selected camera
        for camera_id in c_ids:
            # We are resetting the gradients
            self.optimizer.zero_grad()
            camera = self.cameras[camera_id]
            camera.optimizer.zero_grad()

            # now here we need to update the camera transformation
            camera.update_transform()

            # we here now compute the size of the image
            height, width = camera.image_size

            # compute the pixel coordinates for the batch of rays
            if active_sampling:
                # Here we can modify and change the values rather than 8x8 to select the pixel coordinates
                with torch.no_grad():
                    sub_height = int(height / 8)
                    sub_width = int(width / 8)
                    pixel_u_list = []
                    pixel_v_list = []

                    # Determine the number of samples for each grid cell
                    # and sample the pixel coordinates
                    num_samples = torch.zeros(64, dtype=torch.int32, device=device)
                for i in range(64):
                    # Determine the number of samples for the current grid cell
                    num_samples_i = num_samples[i]

                    # Generate random values for u-coordinate within the grid cell
                    rand_u = torch.rand(num_samples_i) * (sub_width - 1)
                    
                    # Convert u-coordinate to integer and move to device
                    u_coordinates_i = rand_u.to(torch.int16).to(device)

                    # Calculate offset for u-coordinate based on grid cell position
                    u_offset = (i % 8) * sub_width

                    # Adjust u-coordinate based on grid cell position and add to list
                    adjusted_u_coordinates_i = u_coordinates_i + u_offset
                    pixel_u_list.append(adjusted_u_coordinates_i)

                    # Generate random values for v-coordinate within the grid cell
                    rand_v = torch.rand(num_samples_i) * (sub_height - 1)
                    
                    # Convert v-coordinate to integer and move to device
                    v_coordinates_i = rand_v.to(torch.int16).to(device)

                    # Calculate offset for v-coordinate based on grid cell position
                    v_offset = int(i / 8) * sub_height

                    # Adjust v-coordinate based on grid cell position and add to list
                    adjusted_v_coordinates_i = v_coordinates_i + v_offset
                    pixel_v_list.append(adjusted_v_coordinates_i)

                # Concatenate lists to get final pixel coordinates
                u_coord = torch.cat(pixel_u_list)
                v_coord = torch.cat(pixel_v_list)
            else:
                # Use random sampling to select the pixel coordinates
                u_coord = (torch.rand(batch_size) * (width - 1)).to(torch.int16).to(device)
                v_coord = (torch.rand(batch_size) * (height - 1)).to(torch.int16).to(device)

            # Render the batch of rays
            depth, rgb, depth_variance = self.render_rays(u_coord, v_coord, camera)

            # Get the ground truth depth and RGB values
            depth_gt = torch.cat([camera.depth_image[v, u].unsqueeze(0) for u, v in zip(u_coord, v_coord)])
            rgb_gt = torch.cat([camera.rgb_image[v, u, :].unsqueeze(0) for u, v in zip(u_coord, v_coord)])

            # Ignore the depth values for pixels that don't hit any surface
            depth[depth_gt == 0] = 0

            # Compute the inverse variance
            with torch.no_grad():
                depth_var_Sqrt = torch.sqrt(depth_variance)
                inv_var = torch.reciprocal(depth_var_Sqrt)
                inv_var[inv_var.isinf()] = 1
                inv_var[inv_var.isnan()] = 1

            # Compute the loss for the depth and the RGB values
            # Compute depth loss
            depth_difference = torch.abs(depth - depth_gt)
            weighted_depth_difference = depth_difference * inv_var
            depth_loss = torch.mean(weighted_depth_difference)

            # Compute RGB loss
            rgb_difference = torch.abs(rgb - rgb_gt)
            weighted_rgb_difference = 5 * torch.mean(rgb_difference)
            rgb_loss = weighted_rgb_difference

            # Compute total loss
            total_loss = depth_loss + rgb_loss
            #Here the loss and update the parameters
            total_loss.backward()
            self.optimizer.step()

            if camera_id > 0:
                self.cameras[camera_id].optimizer.step()

            # Update the active sampling probabilities
            if active_sampling:
                with torch.no_grad():
                    # Compute depth difference
                    depth_difference = torch.abs(depth - depth_gt)

                    # Compute RGB difference
                    rgb_difference = torch.abs(rgb - rgb_gt)
                    rgb_difference_sum = torch.sum(rgb_difference, dim=1)

                    # Combine depth and RGB differences
                    error = depth_difference + rgb_difference_sum
                    # Compute the mean error for each grid cell
                    # Compute cumulative sum of num_samples
                    num_samples_cumsum = torch.cumsum(num_samples, dim=0)

                    # Initialize active_sampling_probabilities
                    active_sampling_probabilities = torch.zeros(64, device=device)

                    # Compute mean error for the first grid cell
                    mean_error_0 = torch.mean(error[:num_samples_cumsum[0]])
                    active_sampling_probabilities[0] = mean_error_0
                    for i in range(1, 64):
                        # Determine the start and end indices for the error array slice
                        start_index = num_samples_cumsum[i - 1]
                        end_index = num_samples_cumsum[i]
                        
                        # Extract the relevant error values
                        error_slice = error[start_index:end_index]
                        
                        # Compute the mean error for the current grid cell
                        mean_error_i = torch.mean(error_slice)
                        
                        # Update active_sampling_probabilities for the current grid cell
                        active_sampling_probabilities[i] = mean_error_i
                        
                    active_sampling_probabilities_sum = torch.sum(active_sampling_probabilities)
                    normalized_probs = active_sampling_probabilities / active_sampling_probabilities_sum
                    # Update grid_sampling_probs for the current camera
                    self.cameras[camera_id].grid_sampling_probs = normalized_probs
                    
                    
   