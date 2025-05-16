import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Feeder(Dataset):
    def __init__(self, data_path, split="train", train_ratio=0.7, seed=1123,
                 segment_len=2000, stride=2000, label_pad_value=-100, use_mmap=True):
        self.data_path = data_path
        self.segment_len = segment_len
        self.stride = stride
        self.label_pad_value = label_pad_value
        self.use_mmap = use_mmap
                
        all_patients = []
        for group, max_id in [("a", 39), ("b", 42)]:
            for i in range(1, max_id + 1):
                name = f"{group}_patient_{i}"
                data_file = os.path.join(data_path, f"{name}.npy")
                label_file = os.path.join(data_path, f"{name}_vid_lbl.npy")
                if os.path.exists(data_file) and os.path.exists(label_file):
                    all_patients.append((name, data_file, label_file))
                else:
                    print(f"Missing file for {name}, skipped.")

      
        np.random.seed(seed)
        np.random.shuffle(all_patients)
        num_train = int(len(all_patients) * train_ratio)
        selected = all_patients[:num_train] if split == "train" else all_patients[num_train:]

        # legal segment list
        self.segment_indices = []
        for patient_name, data_file, label_file in selected:
            label = np.load(label_file)
            while label.size > 0 and label[-1] == 5:
                label = label[:-1]

            # Label mapping
            label = np.where(np.isin(label, [1, 2, 3]), 1, label)
            label = np.where(label == 4, 2, label)
            T_valid = len(label)

            num_segments = (T_valid - 1) // stride + 1
            for seg_id in range(num_segments):
                start = seg_id * stride
                end = min(start + segment_len, T_valid)
                self.segment_indices.append((patient_name, data_file, label_file, start, end, T_valid))

        print(f"Loaded {len(self.segment_indices)} segments in split '{split}'.")

    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, index):
        patient_name, data_file, label_file, start, end, T_valid = self.segment_indices[index]

        data = np.load(data_file)[:, :T_valid, :, :]
        label = np.load(label_file)[:T_valid]

        # Label mapping
        label = np.where(np.isin(label, [1, 2, 3]), 1, label)
        label = np.where(label == 4, 2, label)

        data = data[:2]  # remove illegal channel

        segment_data = data[:, start:end, :, :]
        segment_label = label[start:end]

        # padding
        pad_len = self.segment_len - segment_data.shape[1]
        if pad_len > 0:
            padded_data = np.zeros((2, self.segment_len, 17, 1), dtype=np.float32)
            padded_data[:, :segment_data.shape[1], :, :] = segment_data
            segment_data = padded_data

            padded_label = np.full((self.segment_len,), self.label_pad_value, dtype=np.int64)
            padded_label[:segment_label.shape[0]] = segment_label
            segment_label = padded_label

        return (
            torch.tensor(segment_data, dtype=torch.float32),       # (2, segment_len, 17, 1)
            torch.tensor(segment_label, dtype=torch.long),         # (segment_len,)
            patient_name
        )