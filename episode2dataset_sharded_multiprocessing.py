import os
import sys
import glob
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json
from enum import Enum
from concurrent.futures import ProcessPoolExecutor

# --- Utility and config ---
class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"

def filter_small_actions(actions, pos_thresh=0.01, rot_thresh=0.06, check_gripper=True):
    actions = np.asarray(actions)
    N = actions.shape[0]
    valid_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        act = actions[i]
        delta_xyz = act[:3]
        delta_euler = act[3:6]
        gripper = act[6] if len(act) > 6 else 0
        pos_movement = np.linalg.norm(delta_xyz)
        rot_movement = np.linalg.norm(delta_euler)
        if pos_thresh is None and rot_thresh is None:
            is_valid = True
        elif pos_thresh is None:
            is_valid = (rot_movement > rot_thresh)
        elif rot_thresh is None:
            is_valid = (pos_movement > pos_thresh)
        else:
            is_valid = (pos_movement > pos_thresh) or (rot_movement > rot_thresh)
        if check_gripper and i > 0 and len(actions[i-1]) > 6 and actions[i - 1][6] != gripper:
            is_valid = True
        valid_mask[i] = is_valid
    return valid_mask

# --- Episode parsing logic (adapted from SFT style) ---
def parse_episode_worker(args):
    ep_file, image_key, apply_action_filter, episode_id = args
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    data = np.load(ep_file, allow_pickle=True)
    if "arr_0" in data:
        data = data["arr_0"].item()
    else:
        data = data.item()
    instruction = data['instruction']
    if isinstance(instruction, np.ndarray):
        instruction = instruction.tolist()[0]
    elif isinstance(instruction, list):
        instruction = instruction[0]
    instruction = str(instruction)
    actions = np.asarray(data["action"])
    states = np.asarray(data["state"])
    images = data["image"]
    if isinstance(images, list):
        images = np.asarray([np.asarray(img) for img in images])
    else:
        images = np.asarray(images)
    if apply_action_filter:
        mask = filter_small_actions(actions)
        actions = actions[mask]
        states = states[mask]
        images = images[mask]
    flat_samples = []
    for i in range(len(actions)):
        if len(images.shape) == 4:
            img = images[i]
        elif len(images.shape) == 3:
            img = images
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}")
        img_tensor = to_tensor(img).float()
        state_tensor = torch.from_numpy(states[i]).float()
        action_tensor = torch.from_numpy(actions[i]).float()
        sample = {
            "observation.state": state_tensor,
            "action": action_tensor,
            image_key: img_tensor,
            "task": instruction,
            "episode_index": episode_id,
            "frame_index": i,
            "timestamp": float(i) / 30.0,
        }
        flat_samples.append(sample)
    return flat_samples

# --- Main sharding logic ---
def main():
    parser = argparse.ArgumentParser(description="Shard .npz episodes into .pt files for pi-zero SFT training (multiprocessing version).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .npz episode files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save sharded .pt files and metadata")
    parser.add_argument("--shard_size", type=int, default=128, help="Number of episodes per shard")
    parser.add_argument("--image_key", type=str, default="observation.images.top", help="Key for images in output samples")
    parser.add_argument("--apply_action_filter", action="store_true", help="Apply action filtering to each episode")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes for parallel episode parsing")
    parser.add_argument("--train_split", type=float, default=0.95, help="Fraction of episodes to use for training (rest for validation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(glob.glob(str(data_dir / "*.npz")))
    num_episodes = len(episode_files)
    # Train/val split in terms of episodes. 
    np.random.seed(args.seed)
    indices = np.arange(num_episodes)
    np.random.shuffle(indices)
    train_count = int(num_episodes * args.train_split)
    train_indices = indices[:train_count]
    val_indices = indices[train_count:]
    train_files = [episode_files[i] for i in train_indices]
    val_files = [episode_files[i] for i in val_indices]
    splits = [("train", train_files, train_indices), ("val", val_files, val_indices)]

    print(f"Found {num_episodes} episodes. Creating {num_episodes} shards.")
    print(f"Using {args.num_workers} worker processes.")

    # For each split, do sharding and save metadata
    for split_name, split_files, split_indices in splits:
        if len(split_files) == 0:
            continue
        num_split_episodes = len(split_files)
        num_split_shards = (num_split_episodes + args.shard_size - 1) // args.shard_size
        print(f"[{split_name}] {num_split_episodes} episodes, {num_split_shards} shards.")
        split_shard_metadata = []
        for shard_idx in tqdm(range(num_split_shards), desc=f"Sharding episodes ({split_name})", total=num_split_shards):
            start = shard_idx * args.shard_size
            end = min((shard_idx + 1) * args.shard_size, num_split_episodes)
            shard_files = split_files[start:end]
            episode_args = [
                (ep_file, args.image_key, args.apply_action_filter, split_indices[start + ep_idx])
                for ep_idx, ep_file in enumerate(shard_files)
            ]
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                results = list(tqdm(
                    executor.map(parse_episode_worker, episode_args),
                    total=len(shard_files),
                    desc=f"Parsing shard {shard_idx}"
                ))
            shard_samples = [sample for samples in results for sample in samples]
            shard_path = shards_dir / f"{split_name}_shard_{shard_idx:05d}.pt"
            torch.save(shard_samples, shard_path)
            split_shard_metadata.append({
                "shard_idx": shard_idx,
                "file": str(shard_path),
                "num_episodes": len(shard_files),
                "num_samples": len(shard_samples),
                "episode_range": [start, end-1],
            })
        # Save split metadata if there are episodes
        if len(split_shard_metadata) > 0:
            meta = {
                "num_shards": num_split_shards,
                "shard_size": args.shard_size,
                "total_episodes": num_split_episodes,
                "shards": split_shard_metadata,
                "image_key": args.image_key,
                "apply_action_filter": args.apply_action_filter,
            }
            json_path = output_dir / f"{split_name}_dataset_sharded.json"
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
    print(f"Sharding complete. Train/val metadata written to {output_dir}")

if __name__ == "__main__":
    main()
    
    """
    python /nvme_data/tonghe/RL4VLA/episode2dataset_sharded_multiprocessing.py \
        --data_dir /nvme_data/tonghe/RL4VLA/datasets/mp_collect/PutOnPlateInScene25Main-v3/12800/data \
            --output_dir /nvme_data/tonghe/RL4VLA/datasets/mp_collect/PutOnPlateInScene25Main-v3/12800/shards \
                --num_workers 16
    """