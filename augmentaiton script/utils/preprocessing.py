import numpy as np
import logging

def center_pose_segment(data_np):
    """Subtracts the mean pose (average over T, V, M) from the segment."""
    # data_np shape: (C, T, V, M) e.g., (2, 1000, 17, 1)
    C, T, V, M = data_np.shape
    if T*V*M == 0: # Handle empty segment
        logging.warning("Attempting to center an empty or zero-dimension segment.")
        return data_np, np.zeros(C, dtype=data_np.dtype)

    # Calculate mean over T, V, M dimensions for each channel (X, Y)
    # Reshape to (C, -1) to safely calculate mean, ignoring potential all-zero frames/joints
    data_flat_for_mean = data_np.reshape(C, -1)
    segment_mean = data_flat_for_mean.mean(axis=1, keepdims=True) # Shape (C, 1)

    # Reshape mean for broadcasting: (C, 1, 1, 1)
    segment_mean_reshaped = segment_mean.reshape(C, 1, 1, 1)
    centered_data = data_np - segment_mean_reshaped
    return centered_data, segment_mean.squeeze() # Return centered data and the mean (C,)

def decenter_pose_segment(centered_data_np, segment_mean):
    """Adds the segment mean back to the centered data."""
    # centered_data_np shape: (C, T, V, M)
    # segment_mean shape: (C,)
    C, T, V, M = centered_data_np.shape
    if T*V*M == 0: return centered_data_np # Return if empty

    if segment_mean.shape[0] != C:
         raise ValueError(f"Shape mismatch: data has {C} channels, but mean has shape {segment_mean.shape}")

    # Add mean back (reshape mean for broadcasting)
    segment_mean_reshaped = segment_mean.reshape(C, 1, 1, 1)
    decentered_data = centered_data_np + segment_mean_reshaped
    return decentered_data