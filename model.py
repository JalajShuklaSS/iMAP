import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import time

#from visualizer import Visualizer
#import mcubes


@torch.jit.script
def mish(x):
    return x * torch.tanh(F.softplus(x))


class IMAP(nn.Module):
    def __init__(self):
        super(IMAP, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Learn a positional embedding from x,y,z to 93 features
        self.positional_embedding = nn.Linear(3, 93, bias=False).to(device)
        nn.init.normal_(self.positional_embedding.weight, 0.0, 25.0)

        # NeRF model with 4 hidden layers of size 256
        self.fc1 = nn.Linear(93,256).to(device)
        self.fc2 = nn.Linear(256,256).to(device)
        self.fc3 = nn.Linear(256+93,256).to(device)
        self.fc4 = nn.Linear(256,256).to(device)
        self.fc5 = nn.Linear(256,4, bias=False).to(device)
        self.fc5.weight.data[3,:]*=0.1


    def forward(self, pos):
        # Position embedding uses a sine activation function
        position_embedding = torch.sin(self.positional_embedding(pos))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # NeRF model
        x = F.relu(self.fc1(position_embedding))
        x = torch.cat([F.relu(self.fc2(x)), position_embedding], dim=1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Output is a 4D vector (r,g,b, density)
        out = self.fc5(x)

        return out


class Camera():
    def __init__(self, rgb_image, depth_image, 
                 position_x, position_y, position_z, 
                 rotation_x, rotation_y, rotation_z, 
                 light_scale=0.0, light_offset=0.0, 
                 focal_length_x=525.0, focal_length_y=525.0, 
                 principal_point_x=319.5, principal_point_y=239.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Camera parameters
        self.params = torch.tensor([rotation_x, rotation_y, rotation_z, 
                                    position_x, position_y, position_z, 
                                    light_scale, light_offset]).detach().requires_grad_(True)
        
        # Camera calibration parameters
        self.focal_length_x = focal_length_x
        self.focal_length_y = focal_length_y
        self.principal_point_x = principal_point_x
        self.principal_point_y = principal_point_y

        # Camera intrinsic matrix and its inverse
        self.K = torch.tensor([
            [focal_length_x, 0.0, principal_point_x],
            [0.0, focal_length_y, principal_point_y],
            [0.0, 0.0, 1.0],
            ], device=device, dtype=torch.float32, requires_grad=False)

        self.K_inverse = torch.tensor([
            [1.0/focal_length_x, 0.0, -principal_point_x/focal_length_x],
            [0.0, 1.0/focal_length_y, -principal_point_y/focal_length_y],
            [0.0, 0.0, 1.0],
            ], device=device, dtype=torch.float32, requires_grad=False)

        # Conversion factor for depth from 16bit color
        self.depth_conversion_factor = 1 / 50000.0

        # RGB and depth images
        self.set_images(rgb_image, depth_image)

        self.exp_a = torch.FloatTensor(1)
        self.rotation_matrix = torch.zeros(3,3, device=device)
        self.translation_matrix = torch.zeros(3,3, device=device)
        self.grid_sampling_probs = torch.full((64,), 1.0/64, device=device)
        self.image_size = depth_image.shape

        # Update transformation matrix
        self.update_transform()

        # Optimizer for camera parameters
        self.optimizer = optim.Adam([self.params], lr=0.005)

    def set_images(self, rgb_image, depth_image):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        self.rgb_image = torch.from_numpy((rgb_image).astype(np.float32)).to(device) / 256.0
        self.depth_image = torch.from_numpy(depth_image.astype(np.float32)).to(device) * self.depth_conversion_factor

    def update_transform(self):
        # Update transformation matrix based on camera parameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        identity_matrix = torch.eye(3, device=device)

        # Create skew symmetric matrices
        skew_symmetric_matrices = [torch.zeros((3,3), device=device) for _ in range(3)]
        skew_symmetric_matrices[0][1, 2] = -1
        skew_symmetric_matrices[0][2, 1] = 1
        skew_symmetric_matrices[1][2, 0] = -1
        skew_symmetric_matrices[1][0, 2] = 1
        skew_symmetric_matrices[2][0, 1] = -1
        skew_symmetric_matrices[2][1, 0] = 1

        # Compute the norm of the rotation vector (gets the angle of rotation)
        rotation_norm = torch.norm(self.params[0:3])

        # Compute the inverse of the rotation norm 
        rotation_norm_inverse = 1.0 / (rotation_norm + 1e-12)

        # Normalize the rotation vector
        rotation_vector = rotation_norm_inverse * self.params[0:3]

        # Compute sine and cosine of the rotation for the rotation matrix
        cos_theta = torch.cos(rotation_norm)
        sin_theta = torch.sin(rotation_norm)

        # Compute the skew symmetric matrix for the rotation vector
        skew_symmetric_matrix = rotation_vector[0]*skew_symmetric_matrices[0] + rotation_vector[1]*skew_symmetric_matrices[1] + rotation_vector[2]*skew_symmetric_matrices[2]

        # Compute the square of the skew symmetric matrix
        skew_symmetric_matrix_squared = torch.matmul(skew_symmetric_matrix, skew_symmetric_matrix)

        # Compute the rotation matrix using the Rodrigues' rotation formula
        rotation_matrix = identity_matrix + sin_theta * skew_symmetric_matrix + (1.0 - cos_theta) * skew_symmetric_matrix_squared

        self.rotation_matrix = rotation_matrix
        self.translation_matrix = self.params[3:6]

        # Compute the exponential of the lighting parameter a
        self.exp_a = torch.exp(self.params[6])


    def rays_for_pixels(self, u, v):
        '''Compute rays for a batch of pixels.'''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = u.shape[0]
        homogeneous_coordinates = torch.ones(batch_size, 3, 1, device=device)
        homogeneous_coordinates[:, 0, 0] = u
        homogeneous_coordinates[:, 1, 0] = v

        # Compute ray vectors in camera coordinates, then rotate to world coordinates
        camera_coords = torch.matmul(self.K_inverse, homogeneous_coordinates)
        ray = torch.matmul(self.rotation_matrix, camera_coords)[:,:,0]

        # Normalize the ray vectors to be length 1 (since really we want the direction of the rays)
        with torch.no_grad():
            ray_length_inverse = 1.0 / torch.norm(ray, dim=1).reshape(batch_size,1).expand(batch_size,3)

        normalized_ray = ray * ray_length_inverse

        # We just copy the translation matrix for each ray in the batch because they all originate 
        # from the same point (the camera center)
        return normalized_ray, self.translation_matrix.reshape(1,3).expand(batch_size,3)

