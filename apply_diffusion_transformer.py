import os
import numpy as np
import torch
import torch.nn as nn
import random
import argparse
from tqdm import tqdm
import shutil
import traceback
import logging 
from diffusers import DDPMScheduler
from collections import OrderedDict
import math
import sys

from diffusion_models.transformer_denoiser import SimplePoseTransformerDenoiser

from utils.preprocessing import center_pose_segment, decenter_pose_segment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_diffusion_transformer_checkpoint(path, device, model_args):
    abs_path = os.path.abspath(path)

    model = SimplePoseTransformerDenoiser(**model_args)
    model.to(device)
    model_weights_loaded_flag = False

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    extracted_state_dict = None

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint: 
            extracted_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: 
            extracted_state_dict = checkpoint['state_dict']
        else: 
            extracted_state_dict = checkpoint
    elif isinstance(checkpoint, OrderedDict) or (hasattr(checkpoint, 'keys') and hasattr(checkpoint, 'values')):
        extracted_state_dict = checkpoint

    if extracted_state_dict and isinstance(extracted_state_dict, dict):
        weights_to_load = OrderedDict([[k.replace('module.', '', 1), v] for k, v in extracted_state_dict.items()])
        missing_keys, unexpected_keys = model.load_state_dict(weights_to_load, strict=False)

        loaded_keys_count = len(set(weights_to_load.keys()).intersection(set(model.state_dict().keys())) - set(missing_keys))

        if loaded_keys_count > 0:
            model_weights_loaded_flag = True

    model.eval()
    return model, model_weights_loaded_flag

@torch.no_grad()
def augment_pose_segment_diffusion_transformer(pose_segment_np, model, noise_scheduler, t_aug_start, device, num_inference_steps=50, generator=None):
    model.eval()

    C, T, V, M = pose_segment_np.shape
    F = C * V * M

    if T == 0: 
        return pose_segment_np

    centered_data_np, segment_mean = center_pose_segment(pose_segment_np)
    pose_sequence = centered_data_np.transpose(1, 0, 2, 3).reshape(T, F)
    pose_tensor = torch.from_numpy(pose_sequence).unsqueeze(0).float().to(device)

    noise = torch.randn(pose_tensor.shape, generator=generator, device=device, dtype=pose_tensor.dtype)
    start_timesteps = torch.tensor([t_aug_start], device=device, dtype=torch.long)
    noisy_pose = noise_scheduler.add_noise(pose_tensor.to(device), noise, start_timesteps)

    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    try:
        valid_indices = torch.where(noise_scheduler.timesteps <= t_aug_start)[0]
        start_index = valid_indices[0].item() if len(valid_indices) > 0 else 0
        inference_timesteps = noise_scheduler.timesteps[start_index:]
    except: 
        inference_timesteps = noise_scheduler.timesteps

    current_pose = noisy_pose.to(device)
    for t_step in inference_timesteps:
        timestep_tensor = torch.tensor([t_step], device=device, dtype=torch.long)
        noise_pred = model(current_pose, timestep_tensor) # Denoise model call
        scheduler_output = noise_scheduler.step(noise_pred, t_step, current_pose, generator=generator, return_dict=True)
        current_pose = scheduler_output.prev_sample

    augmented_centered_sequence = current_pose.squeeze(0).cpu().numpy()
    augmented_centered_data = augmented_centered_sequence.reshape(T, C, V, M).transpose(1, 0, 2, 3)          # Reshape to original format
    augmented_denorm_np = decenter_pose_segment(augmented_centered_data, segment_mean)
    return augmented_denorm_np.astype(np.float32)

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    transformer_model_args = { 'input_features': 2*17*1, 'sequence_length': args.segment_len, 'model_dim': args.model_dim, 'num_layers': args.num_layers, 'num_heads': args.num_heads, 'dim_feedforward': args.dim_feedforward, 'dropout': args.dropout }
    diffusion_model, model_weights_actually_loaded = load_diffusion_transformer_checkpoint(args.diffusion_checkpoint, device, transformer_model_args)

    apply_diffusion = (diffusion_model is not None) and model_weights_actually_loaded

    # --- Print final decision on apply_diffusion ---
    print(f"--model_weights_actually_loaded = {model_weights_actually_loaded}")
    print(f"--args.diffusion_checkpoint = {args.diffusion_checkpoint}")
    print(f"--Final apply_diffusion flag = {apply_diffusion}")
    # ---

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    source_dir = args.source_dir; target_dir = args.target_dir
    num_diffusion = args.num_augmentations if apply_diffusion else 0
    t_augs = []
    if apply_diffusion and num_diffusion > 0:
         min_t = max(1, int(args.t_range[0]))
         max_t = min(noise_scheduler.config.num_train_timesteps -1, int(args.t_range[1]))

         if min_t >= max_t: 
            max_t = min_t + 1

         t_augs = np.linspace(min_t, max_t, num_diffusion, dtype=int)
         print(f"Generating {num_diffusion} diffusion versions using t_aug: {t_augs}")

    os.makedirs(target_dir, exist_ok=True)
    all_segment_files = [f for f in os.listdir(source_dir) if f.endswith('.npy') and '_lbl' not in f]
    logging.info(f"Found {len(all_segment_files)} source segments in {source_dir}") # Use logging for general info

    diffusion_count, copied_count, error_count = 0, 0, 0
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    for data_filename in tqdm(all_segment_files, desc="Applying Diffusion"):
        base_name = data_filename.replace('.npy', '')
        data_filepath = os.path.join(source_dir, data_filename)
        label_filepath = os.path.join(source_dir, f"{base_name}_lbl.npy")
        if not os.path.exists(label_filepath): continue

        try:
            target_data_file_src = os.path.join(target_dir, f"{base_name}.npy")
            target_label_file_src = os.path.join(target_dir, f"{base_name}_lbl.npy")
            if not os.path.exists(target_data_file_src) or args.overwrite: shutil.copy2(data_filepath, target_data_file_src); copied_count += 1
            if not os.path.exists(target_label_file_src) or args.overwrite: shutil.copy2(label_filepath, target_label_file_src)

            if apply_diffusion and diffusion_model and len(t_augs) > 0:
                segment_data_np = np.load(data_filepath)
                if segment_data_np.shape[1] != args.segment_len:
                    continue

                for i, t_aug in enumerate(t_augs):
                     augmented_data_np = augment_pose_segment_diffusion_transformer(
                         segment_data_np, model=diffusion_model, noise_scheduler=noise_scheduler,
                         t_aug_start=t_aug, device=device, num_inference_steps=args.diff_steps,
                         generator=generator
                     )
                     diff_name = f"{base_name}_diff{i}_t{t_aug}"
                     target_data_file_aug = os.path.join(target_dir, f"{diff_name}.npy")
                     target_label_file_aug = os.path.join(target_dir, f"{diff_name}_lbl.npy")
                     np.save(target_data_file_aug, augmented_data_np.astype(np.float32))
                     if os.path.exists(target_label_file_src): shutil.copy2(target_label_file_src, target_label_file_aug)
                     diffusion_count += 1
        except Exception as e:
            error_count += 1
            logging.error(f"ERROR on {data_filename}: {e}")
            traceback.print_exc()

    print(f"\nDiffusion augmentation finished.")
    print(f"  Source segments processed/copied: {len(all_segment_files)}")
    print(f"  Diffusion segments saved: {diffusion_count}")

    if error_count > 0: 
        print(f"  Segments failed: {error_count}")

# --- if __name__ == "__main__": block (as before) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply diffusion augmentation (Transformer + Centering) to segments.")
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--target-dir', required=True)
    parser.add_argument('--diffusion-checkpoint', default=None)
    parser.add_argument('--segment-len', type=int, required=True)
    parser.add_argument('--num-augmentations', type=int, default=1)
    parser.add_argument('--t-range', type=float, nargs=2, default=[150, 400])
    parser.add_argument('--diff-steps', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    # --- Transformer Architecture Args ---
    parser.add_argument('--model-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dim-feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    # Construct model_args dictionary AFTER parsing
    transformer_arg_names = ['model_dim', 'num_layers', 'num_heads', 'dim_feedforward', 'dropout']
    args.model_args = {k: getattr(args, k) for k in transformer_arg_names if hasattr(args, k)}
    args.model_args['input_features'] = 2 * 17 * 1
    args.model_args['sequence_length'] = args.segment_len
    main(args)