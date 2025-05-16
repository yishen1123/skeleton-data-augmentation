# preprocess_center_data.py
import os
import numpy as np
import argparse
from tqdm import tqdm
import traceback
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Make sure preprocessing functions are accessible ---
try:
    # Assuming utils/preprocessing.py is importable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.join(os.path.dirname(script_dir), 'utils')
    sys.path.insert(0, os.path.dirname(script_dir))
    from utils.preprocessing import center_pose_segment
except ImportError:
    logging.error("Could not import center_pose_segment from utils.preprocessing.")
    # Define dummy function
    def center_pose_segment(data_np):
        logging.warning("Using Dummy Centering")
        return data_np, np.zeros(data_np.shape[0])
    # exit(1) # Exit in real use case

def main(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    os.makedirs(target_dir, exist_ok=True)

    all_segment_files = [f for f in os.listdir(source_dir) if f.endswith('.npy') and '_lbl' not in f]
    logging.info(f"Found {len(all_segment_files)} segments in {source_dir} to preprocess.")

    error_count = 0
    for data_filename in tqdm(all_segment_files, desc="Centering Segments"):
        data_filepath = os.path.join(source_dir, data_filename)
        target_filepath = os.path.join(target_dir, data_filename)

        # if os.path.exists(target_filepath) and not args.overwrite: continue

        try:
            segment_data_np = np.load(data_filepath)
            # Apply centering
            centered_data_np, _ = center_pose_segment(segment_data_np)
            # Save centered data
            np.save(target_filepath, centered_data_np.astype(np.float32))
        except Exception as e:
            error_count += 1; logging.error(f"ERROR processing {data_filename}: {e}")
            if error_count > 20: logging.error("Too many errors, stopping."); break

    logging.info(f"Centering finished. Centered segments saved: {len(all_segment_files) - error_count}")
    if error_count > 0: logging.info(f"Segments failed: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Center pre-segmented pose data.")
    parser.add_argument('--source-dir', required=True, help='Directory with segments (e.g., split_data_tfm)')
    parser.add_argument('--target-dir', required=True, help='Directory to save centered segments')
    # parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    main(args)