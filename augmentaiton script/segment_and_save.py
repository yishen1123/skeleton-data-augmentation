# segment_and_save.py
# Uses the provided Feeder class to generate and save individual segments.
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader # Use DataLoader for iteration
import random
import argparse
from tqdm import tqdm
import traceback

from feeders.my_feeder import Feeder

def main(args):
    target_dir = args.target_dir
    os.makedirs(target_dir, exist_ok=True)
    feeder_args = {
        'data_path': args.source_dir,
        'split': args.split,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'segment_len': args.segment_len,
        'stride': args.stride,
        'label_pad_value': args.label_pad_value,
    }

    print(f"Initializing Feeder for split '{args.split}' to generate segments...")

    dataset = Feeder(**feeder_args)

    num_segments_to_process = len(dataset)
    print(f"Feeder initialized. Found {num_segments_to_process} segments to process and save.")

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    saved_count = 0
    error_count = 0
    for i, batch_data in enumerate(tqdm(data_loader, desc=f"Saving Segments ({args.split})")):
        segment_data_tensor, segment_label_tensor, patient_name_list = batch_data

        # Squeeze the batch dimension and convert back to numpy for saving
        segment_data_np = segment_data_tensor.squeeze(0).numpy()
        segment_label_np = segment_label_tensor.squeeze(0).numpy()
        # Get patient name
        patient_name = patient_name_list[0]
        if isinstance(patient_name, (list, tuple)):
            patient_name = patient_name[0]

        segment_id = f"{patient_name}_seg{i}"

        target_data_file = os.path.join(target_dir, f"{segment_id}.npy")
        target_label_file = os.path.join(target_dir, f"{segment_id}_lbl.npy")

        np.save(target_data_file, segment_data_np.astype(np.float32)) 
        np.save(target_label_file, segment_label_np.astype(np.int64)) 
        saved_count += 1

    print(f"\nSegmentation and saving finished for split '{args.split}'.")
    print(f"  Total segments successfully saved: {saved_count} / {num_segments_to_process}")
    if error_count > 0:
        print(f"  Number of segments failed: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment data using existing Feeder and save segments.")
    parser.add_argument('--source-dir', required=True, help='Dir with original patient .npy files')
    parser.add_argument('--target-dir', required=True, help='Dir to save individual segment files')
    parser.add_argument('--split', default='train', choices=['train', 'test'], help='Which split to process')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=1123)
    parser.add_argument('--segment-len', type=int, default=2000)
    parser.add_argument('--stride', type=int, default=2000)  
    parser.add_argument('--label-pad-value', type=int, default=-100)

    args = parser.parse_args()
    main(args)