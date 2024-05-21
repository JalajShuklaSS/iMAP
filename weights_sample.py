
import torch
class Weightes_sample():
    def __init__(self) -> None:
        pass

    @staticmethod
    def weighted_samples(bin_positions, bin_weights, num_samples):
        """
        Perform hierarchical sampling of bins using inverse transform sampling.

        Parameters:
        bin_positions: The positions of the bins.
        bin_weights: The weights of the bins.
        num_samples: The number of samples to draw.

        Returns:
        samples: The sampled positions.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Add a small constant to weights to avoid division by zero
        bin_weights = bin_weights + 1e-5

        # Here we are getting the probability density function (pdf) from the weights
        # We start by firtst computing the sum of bin weights along the last dimension
        sum_bin_weights = torch.sum(bin_weights, -1, keepdim=True)

        # Here we are computing the reciprocal of the sum of bin weights
        reciprocal_sum_bin_weights = torch.reciprocal(sum_bin_weights)

        # Here we need to get the probability density function (pdf)
        pdf = bin_weights * reciprocal_sum_bin_weights

        # Compute the cumulative distribution function (cdf) from the pdf
        cdf = torch.cumsum(pdf, -1)

        # Append a zero at the beginning of the cdf
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

        # Generate random numbers for sampling
        random_numbers = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(device)

        # Flatten the random numbers tensor for the searchsorted operation
        random_numbers = random_numbers.contiguous()

        # Find the indices where the random numbers would be inserted to maintain the order of the cdf
        indices = torch.searchsorted(cdf, random_numbers, right=True)

        # Clamp the indices to the valid range
        indices_below = torch.max(torch.zeros_like(indices-1), indices-1)
        indices_above = torch.min((cdf.shape[-1]-1) * torch.ones_like(indices), indices)

        # Stack the indices
        indices_grouped = torch.stack([indices_below, indices_above], -1)

        # Expand the cdf and bin_positions tensors to match the shape of indices_grouped
        
        # So forthat we get the shape of indices_grouped
        num_rows = indices_grouped.shape[0]
        num_cols = indices_grouped.shape[1]
        num_bins = cdf.shape[-1]
        matched_shape = [num_rows, num_cols, num_bins]

        # Now basically expand cdf to match the shape of indices_grouped
        expanded_cdf = cdf.unsqueeze(1).expand(matched_shape)

        # Gather values from cdf based on indices_grouped
        cdf_grouped = torch.gather(expanded_cdf, 2, indices_grouped)

        # Expand bin_positions to match the shape of indices_grouped
        expanded_bin_positions = bin_positions.unsqueeze(1).expand(matched_shape)

        # Gather values from bin_positions based on indices_grouped
        bin_positions_grouped = torch.gather(expanded_bin_positions, 2, indices_grouped)
        
        # Compute the interpolation weights
        denominator = (cdf_grouped[...,1]-cdf_grouped[...,0])
        denominator = torch.where(denominator<1e-5, torch.ones_like(denominator), denominator)
        interpolation_weights = (random_numbers-cdf_grouped[...,0])/denominator

        # Interpolate the samples
        # Extract bin positions for interpolation
        start_bin_positions = bin_positions_grouped[..., 0]
        end_bin_positions = bin_positions_grouped[..., 1]

        # Compute interpolated samples
        interpolated_diff = end_bin_positions - start_bin_positions
        interpolation_weights_expanded = interpolation_weights.unsqueeze(-1)
        samples = start_bin_positions + interpolation_weights_expanded * interpolated_diff
        # Concatenate the samples with the bin_positions and sort them
        samples_concatenated, _ = torch.sort(torch.cat([samples, bin_positions], -1), dim=-1)

        return samples_concatenated
