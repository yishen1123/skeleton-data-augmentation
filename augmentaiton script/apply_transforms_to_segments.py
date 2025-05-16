import os
import numpy as np
import random
import argparse
from tqdm import tqdm
import shutil
import traceback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Augmentation Functions (Keep as before) ---
def random_rotate_segment_unified_center(data_np, max_angle_deg=15.0):
    C, T, V, M = data_np.shape

    if T*V*M == 0: 
        return data_np

    angle_deg = random.uniform(-max_angle_deg, max_angle_deg)
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    center = data_np.reshape(C, -1).mean(axis=1)

    data_flat = data_np.reshape(C, -1).T
    data_rotated = (data_flat - center) @ rot_matrix.T + center
    return data_rotated.T.reshape(C, T, V, M)

def random_scale_segment_unified_center(data_np, scale_range=(0.9, 1.1)):
    C, T, V, M = data_np.shape

    if T*V*M == 0: 
        return data_np

    scale_factor = random.uniform(scale_range[0], scale_range[1])
    center = data_np.reshape(C, -1).mean(axis=1)

    data_flat = data_np.reshape(C, -1).T
    data_scaled = (data_flat - center) * scale_factor + center
    return data_scaled.T.reshape(C, T, V, M)

def random_translate_segment(data_np, max_translate_fraction=0.05):
     C, T, V, M = data_np.shape

     if T*V*M == 0: 
        return data_np

     std_dev = data_np.std() + 1e-6

     offset_x = random.uniform(-max_translate_fraction, max_translate_fraction) * std_dev
     offset_y = random.uniform(-max_translate_fraction, max_translate_fraction) * std_dev

     data_translated = data_np.copy()
     data_translated[0, ...] += offset_x
     data_translated[1, ...] += offset_y
     return data_translated

def add_gaussian_noise_segment(data_np, noise_std_fraction=0.01):
    if data_np.size == 0: 
        return data_np

    std_dev = data_np.std() + 1e-6  # Avoid multiple 0

    noise_sigma = std_dev * noise_std_fraction
    noise = np.random.normal(loc=0, scale=noise_sigma, size=data_np.shape).astype(np.float32)
    return data_np + noise

AUGMENTATION_FUNCTIONS = {"rot": random_rotate_segment_unified_center, "scl": random_scale_segment_unified_center, "trl": random_translate_segment, "noi": add_gaussian_noise_segment,}

def main(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    base_num_augmentations = args.base_num_augmentations # Base augmentations if no specific rule matches

    # Example: {0: 1.0, 1: 5.0, 2: 2.5} means class 0 gets 1*base class 1 gets 5*base, class 2 gets 2.5*base
    class_augment_multipliers = {}
    if args.class_multipliers:
        for item in args.class_multipliers:
            parts = item.split(':')
            if len(parts) == 2:
                try:
                    class_idx = int(parts[0])
                    multiplier = float(parts[1])
                    class_augment_multipliers[class_idx] = multiplier
                except ValueError:
                    logging.warning(f"Could not parse class_multiplier: {item}. Skipping.")
            else:
                logging.warning(f"Invalid format for class_multiplier: {item}. Expected 'class_idx:multiplier'. Skipping.")
    logging.info(f"Using class augmentation multipliers: {class_augment_multipliers}")

    os.makedirs(target_dir, exist_ok=True)
    all_segment_files = [f for f in os.listdir(source_dir) if f.endswith('.npy') and '_lbl' not in f]
    logging.info(f"Found {len(all_segment_files)} segment data files in {source_dir} to process.")

    transform_count = 0
    copied_count = 0
    error_count = 0

    for data_filename in tqdm(all_segment_files, desc="Applying Targeted Transforms"):
        base_name = data_filename.replace('.npy', '')
        data_filepath = os.path.join(source_dir, data_filename)
        label_filepath = os.path.join(source_dir, f"{base_name}_lbl.npy")
        if not os.path.exists(label_filepath): continue

        try:
            segment_data = np.load(data_filepath)
            segment_label_padded = np.load(label_filepath)

            # Copy Original Segment
            target_data_file_orig = os.path.join(target_dir, f"{base_name}.npy")
            target_label_file_orig = os.path.join(target_dir, f"{base_name}_lbl.npy")
            if not os.path.exists(target_data_file_orig) or args.overwrite:
                 shutil.copy2(data_filepath, target_data_file_orig)
                 copied_count +=1
            if not os.path.exists(target_label_file_orig) or args.overwrite:
                 shutil.copy2(label_filepath, target_label_file_orig)

            # Determine number of augmentations for this segment
            valid_labels_in_seg = segment_label_padded[segment_label_padded != args.label_pad_value]
            num_augs_this_segment = base_num_augmentations

            if valid_labels_in_seg.size > 0:
                 unique_labels_in_valid = set(np.unique(valid_labels_in_seg))
                 current_max_multiplier = 1.0 # Default multiplier
                 for cls_idx, multiplier in class_augment_multipliers.items():
                     if cls_idx in unique_labels_in_valid:
                         current_max_multiplier = max(current_max_multiplier, multiplier)
                 num_augs_this_segment = int(base_num_augmentations * current_max_multiplier)

            # Generate and Save Augmented Versions
            for j in range(num_augs_this_segment):
                augmented_data = segment_data.copy()
                applied_augs_desc = []
                # Apply transforms (probabilistically)
                if args.apply_rotate and random.random() < args.aug_prob: 
                    augmented_data = AUGMENTATION_FUNCTIONS["rot"](augmented_data, args.max_angle)
                    applied_augs_desc.append("r")

                if args.apply_scale and random.random() < args.aug_prob: 
                    augmented_data = AUGMENTATION_FUNCTIONS["scl"](augmented_data, tuple(args.scale_range))
                    applied_augs_desc.append("s")

                if args.apply_translate and random.random() < args.aug_prob: 
                    augmented_data = AUGMENTATION_FUNCTIONS["trl"](augmented_data, args.max_translate)
                    applied_augs_desc.append("t")

                if args.apply_noise and random.random() < args.aug_prob: 
                    augmented_data = AUGMENTATION_FUNCTIONS["noi"](augmented_data, args.noise_std)
                    applied_augs_desc.append("n")

                if not applied_augs_desc and num_augs_this_segment > 0 : # If no transform applied but we want to save an aug
                    aug_type_str = f"copy{j}" # Mark as a copy if no actual transform
                elif applied_augs_desc:
                    aug_type_str = "".join(sorted(list(set(applied_augs_desc)))) + str(j)
                else: 
                    continue 

                aug_name = f"{base_name}_tfm_{aug_type_str}"
                target_data_file_aug = os.path.join(target_dir, f"{aug_name}.npy")
                target_label_file_aug = os.path.join(target_dir, f"{aug_name}_lbl.npy")
                np.save(target_data_file_aug, augmented_data.astype(np.float32))
                if os.path.exists(target_label_file_orig): 
                    shutil.copy2(target_label_file_orig, target_label_file_aug)
                transform_count += 1
        except Exception as e:
            error_count += 1
            logging.error(f"ERROR processing segment {data_filename}: {e}"); traceback.print_exc()

    print(f"\nTransform augmentation finished.")
    print(f"  Original segments copied/processed: {copied_count} / {len(all_segment_files)}")
    print(f"  Transform-augmented segments saved: {transform_count}")
    if error_count > 0: print(f"  Number of segments failed: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply targeted traditional transforms to pre-segmented files.")
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--target-dir', required=True)
    parser.add_argument('--base-num-augmentations', type=int, default=1, help='Base number of augmented versions per segment (for majority or non-targeted classes)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--label-pad-value', type=int, default=-100)
    # Augmentation control
    parser.add_argument('--aug-prob', type=float, default=0.5, help="Probability to apply each enabled transform")
    parser.add_argument('--apply-rotate', action='store_true')
    parser.add_argument('--max-angle', type=float, default=10.0)
    parser.add_argument('--apply-scale', action='store_true')
    parser.add_argument('--scale-range', type=float, nargs=2, default=[0.95, 1.05])
    parser.add_argument('--apply-translate', action='store_true'); parser.add_argument('--max-translate', type=float, default=0.03)
    parser.add_argument('--apply-noise', action='store_true'); parser.add_argument('--noise-std', type=float, default=0.005)
    # --- Arguments for targeted augmentation based on class multipliers ---
    parser.add_argument('--class-multipliers', type=str, nargs='*',
                        help="List of 'class_idx:multiplier' strings. E.g., '0:1.0' '1:5.0' '2:2.0'")

    args = parser.parse_args()
    main(args)