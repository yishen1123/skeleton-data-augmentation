# train_diffusion_transformer.py
# Assumes preprocessing (centering) is done beforehand.
# Requires actual model definition and preprocessing functions to be importable.

import argparse
import os
import numpy as np
import torch
import torch.nn as nn # Keep nn for model definition reference if needed below
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from pathlib import Path
import yaml
import logging
from collections import OrderedDict
import math
import traceback # Keep for error reporting
import sys # For sys.exit
import time

from diffusion_models.transformer_denoiser import SimplePoseTransformerDenoiser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger(__name__, log_level="INFO") # Use accelerate logger

class PoseSegmentDataset(Dataset):
    def __init__(self, data_dir, segment_len=500):
        self.data_dir = data_dir
        self.segment_len = segment_len
        self.expected_features = 2 * 17 * 1 # C*V*M

        try:
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy') and '_lbl' not in f]
            if not self.file_list: raise FileNotFoundError("No data files found")
        except FileNotFoundError: raise FileNotFoundError(f"Data directory not found or empty: {data_dir}")
        except Exception as e: raise RuntimeError(f"Error listing files in {data_dir}: {e}") from e
        logger.info(f"Dataset: Found {len(self.file_list)} data segments.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filepath = os.path.join(self.data_dir, self.file_list[index])

        # Data should already be CENTERED and padded (T=segment_len)
        data_np = np.load(filepath).astype(np.float32)

        # Verify shape before reshape
        C, T, V, M = data_np.shape
        current_features = C * V * M
        if T != self.segment_len or current_features != self.expected_features:
                logging.warning(f"Skipping {self.file_list[index]}: Shape mismatch. Expected T={self.segment_len}, F={self.expected_features}. Got C={C},T={T},V={V},M={M}")
                return torch.zeros((self.segment_len, self.expected_features), dtype=torch.float32) # Return dummy

        # Reshape to (T, F)
        data_sequence = data_np.transpose(1, 0, 2, 3).reshape(self.segment_len, self.expected_features)
        return torch.tensor(data_sequence, dtype=torch.float32)

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Diffusion Model")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./diffusion_output")
    parser.add_argument("--segment_len", type=int, default=500)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training Params
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95); parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6); parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_epochs", type=int, default=25)
    # Scheduler Params
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    args = parser.parse_args()
    args.input_features = 2 * 17 * 1
    return args

# Main Training Function
def main():
    args = parse_args()

    # Initialize Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision, log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"), kwargs_handlers=[ddp_kwargs]
    )
    logger.info(accelerator.state, main_process_only=False)
    if args.seed is not None: set_seed(args.seed)

    model_args_dict = { 'input_features': args.input_features, 'sequence_length': args.segment_len, 'model_dim': args.model_dim, 'num_layers': args.num_layers, 'num_heads': args.num_heads, 'dim_feedforward': args.dim_feedforward, 'dropout': args.dropout }
    model = SimplePoseTransformerDenoiser(**model_args_dict)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_schedule)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    train_dataset = PoseSegmentDataset(args.train_data_dir, segment_len=args.segment_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # LR Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps*args.gradient_accumulation_steps, num_training_steps=args.max_train_steps*args.gradient_accumulation_steps)

    # Prepare with Accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare( model, optimizer, train_dataloader, lr_scheduler )

    # --- Trackers ---
    if accelerator.is_main_process:
        run_name = Path(args.output_dir).name
        accelerator.init_trackers(run_name, config=vars(args))

    # --- Training Loop ---
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accum steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(train_dataloader):
            clean_poses = batch # Shape (B, T, F)
            noise = torch.randn(clean_poses.shape, device=accelerator.device)
            bsz = clean_poses.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device, dtype=torch.long)
            noisy_poses = noise_scheduler.add_noise(clean_poses, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_poses, timesteps)
                loss = F.mse_loss(noise_pred, noise)

                if torch.isnan(loss): 
                    logger.warning(f"NaN loss step {global_step}")
                    optimizer.zero_grad()
                    continue

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1); global_step += 1
                current_lr = lr_scheduler.get_last_lr()[0]
                logs = {"step_loss": loss.item(), "lr": current_lr}; progress_bar.set_postfix(**logs)
                accelerator.log({"train_loss_step": loss.item(), "lr": current_lr}, step=global_step)

                # Save checkpoints
                if accelerator.is_main_process:
                    save_checkpoint = False
                    if args.save_steps > 0 and global_step % args.save_steps == 0: 
                        save_checkpoint = True

                    is_last_step = (step + 1 == len(train_dataloader)) or (step + 1 == num_update_steps_per_epoch)

                    if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0 and is_last_step: 
                        save_checkpoint = True
                    if global_step >= args.max_train_steps: 
                        save_checkpoint = True
                    if save_checkpoint:
                        save_path_base = Path(args.output_dir)
                        # Save only the model weights for easier loading later
                        unwrapped_model = accelerator.unwrap_model(model)
                        model_save_path = save_path_base / f"transformer_denoiser_step_{global_step}.pt"
                        # Use accelerator.save for potentially distributed saving
                        accelerator.save(unwrapped_model.state_dict(), model_save_path)
                        logger.info(f"Saved model checkpoint to {model_save_path}")

            if global_step >= args.max_train_steps: 
                break

        progress_bar.close()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process: 
            accelerator.log({"train_loss_epoch": train_loss / len(train_dataloader)}, step=epoch)

        if global_step >= args.max_train_steps: 
            break

    accelerator.end_training()
    logger.info("Training finished!")

if __name__ == "__main__":
    args = parse_args()
    main()