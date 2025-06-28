import os
import glob
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse


def filter_small_actions(actions, pos_thresh=0.01, rot_thresh=0.06, check_gripper=True):
    """Filter out small/insignificant actions to reduce dataset noise."""
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

        # Preserve gripper toggle events
        if check_gripper and i > 0 and len(actions[i-1]) > 6 and actions[i - 1][6] != gripper:
            is_valid = True

        valid_mask[i] = is_valid

    return valid_mask


def compute_dataset_stats(all_states, all_actions, all_images, image_key):
    """
    Compute dataset statistics for pi-zero normalization.
    
    Pi-zero expects dataset_stats in this format:
    {
        "observation.state": {"mean": tensor, "std": tensor},
        "action": {"mean": tensor, "std": tensor},
        "observation.images.image": {"mean": tensor, "std": tensor},  # per-channel
    }
    """
    stats = {}
    
    # State statistics
    if all_states:
        states_tensor = torch.stack(all_states, dim=0)  # [total_steps, state_dim]
        stats["observation.state"] = {
            "mean": states_tensor.mean(dim=0),  # [state_dim]
            "std": states_tensor.std(dim=0),    # [state_dim]
        }
    
    # Action statistics  
    if all_actions:
        actions_tensor = torch.stack(all_actions, dim=0)  # [total_steps, action_dim]
        stats["action"] = {
            "mean": actions_tensor.mean(dim=0),  # [action_dim]
            "std": actions_tensor.std(dim=0),    # [action_dim]
        }
    
    # Image statistics (per-channel)
    if all_images:
        images_tensor = torch.stack(all_images, dim=0)  # [total_steps, 3, H, W]
        # Compute mean and std per channel
        stats[image_key] = {
            "mean": images_tensor.mean(dim=(0, 2, 3), keepdim=True),  # [3, 1, 1]
            "std": images_tensor.std(dim=(0, 2, 3), keepdim=True),    # [3, 1, 1]
        }
    
    return stats


class PiZeroDatasetBuilder:
    """PyTorch dataset builder for pi-zero in RLDS style (episode-based structure)."""
    
    def __init__(self, 
                 data_configs: List[Dict[str, Any]], 
                 image_key: str = "observation.images.image",
                 apply_action_filter: bool = True,
                 success_threshold: int = 6,
                 compute_stats: bool = True):
        """
        Args:
            data_configs: List of data source configs, each containing:
                - "path": path to data directory
                - "compressed": whether files are .npz (True) or other format
                - "filter": whether to apply action filtering
            image_key: Key name for images in the output dataset
            apply_action_filter: Whether to filter small actions
            success_threshold: Number of consecutive successes before truncating episode
            compute_stats: Whether to compute dataset statistics for normalization
        """
        self.data_configs = data_configs
        self.image_key = image_key
        self.apply_action_filter = apply_action_filter
        self.success_threshold = success_threshold
        self.compute_stats = compute_stats
        self.to_tensor = transforms.ToTensor()
    
    def _parse_episode(self, episode_path: str, compressed: bool, use_filter: bool) -> Dict[str, Any]:
        """Parse a single episode file into pi-zero format."""
        
        if compressed:
            data = np.load(episode_path, allow_pickle=True)["arr_0"].item()
        else:
            data = np.load(episode_path, allow_pickle=True).item()
        
        # Extract instruction
        instruction = data['instruction']
        if isinstance(instruction, np.ndarray):
            instruction = instruction.tolist()[0]
        elif isinstance(instruction, list):
            instruction = instruction[0]
        instruction = str(instruction)
        
        # Extract actions, states, images
        actions = np.asarray(data["action"])
        states = np.asarray(data["state"])
        images = data["image"]
        
        # Convert images to numpy arrays if needed
        if isinstance(images, list):
            images = np.asarray([np.asarray(img) for img in images])
        else:
            images = np.asarray(images)
        
        # Apply action filtering if requested
        num_filtered = 0
        if use_filter and self.apply_action_filter:
            mask = filter_small_actions(actions)
            actions = actions[mask]
            states = states[mask]
            images = images[mask]
            num_filtered = mask.shape[0] - mask.sum()
            print(f"Filtered {num_filtered}/{mask.shape[0]} actions from {episode_path}")
        
        # Handle success-based episode truncation (if info available)
        if "info" in data:
            success_count = 0
            truncate_idx = len(actions)
            for i in range(len(actions)):
                if data["info"][i].get("success", False):
                    success_count += 1
                else:
                    success_count = 0
                
                if success_count >= self.success_threshold:
                    truncate_idx = i + 1
                    break
            
            actions = actions[:truncate_idx]
            states = states[:truncate_idx]
            images = images[:truncate_idx]
        
        # Build episode steps in pi-zero format
        steps = []
        episode_states = []
        episode_actions = []
        episode_images = []
        
        for i in range(len(actions)):
            # Handle single image vs multiple images
            if len(images.shape) == 4:  # [T, H, W, C]
                img = images[i]
            elif len(images.shape) == 3:  # Single image [H, W, C] repeated for all steps
                img = images
            else:
                raise ValueError(f"Unexpected image shape: {images.shape}")
            
            # Convert to tensor and ensure correct format [C, H, W] in range [0, 1]
            img_tensor = self.to_tensor(img).float()
            state_tensor = torch.from_numpy(states[i]).float()
            action_tensor = torch.from_numpy(actions[i]).float()
            
            step = {
                "observation.state": state_tensor,
                "action": action_tensor,
                self.image_key: img_tensor,
                "task": instruction,  # pi-zero expects "task" not "language_instruction"
            }
            steps.append(step)
            
            # Collect for statistics computation
            if self.compute_stats:
                episode_states.append(state_tensor)
                episode_actions.append(action_tensor)
                episode_images.append(img_tensor)
        
        # Create episode sample
        episode_sample = {
            "steps": steps,
            "episode_metadata": {
                "file_path": episode_path,
                "episode_length": len(steps),
                "num_filtered_actions": num_filtered,
                "instruction": instruction,
            }
        }
        
        return episode_sample, num_filtered, episode_states, episode_actions, episode_images
    
    def build_dataset(self, 
                     output_file: str, 
                     train_episodes: int = 70, 
                     val_episodes: int = 5,
                     spare_episodes: int = 5) -> Dict[str, Any]:
        """
        Build the complete dataset with train/val splits.
        
        Args:
            output_file: Path to save the dataset
            train_episodes: Number of episodes for training
            val_episodes: Number of episodes for validation  
            spare_episodes: Number of episodes to hold out
        """
        
        # Collect all episode files
        all_files = []
        for config in self.data_configs:
            path = Path(config["path"])
            compressed = config.get("compressed", True)
            use_filter = config.get("filter", True)
            
            if compressed:
                files = sorted(glob.glob(str(path / "*.npz")))
            else:
                files = sorted(glob.glob(str(path / "*.npy")))
            
            print(f"Found {len(files)} files in {path}")
            
            # Reserve some episodes and split train/val
            if spare_episodes > 0:
                files = files[:-spare_episodes]
            
            train_files = files[:train_episodes]
            val_files = files[train_episodes:train_episodes + val_episodes]
            
            print(f"Using {len(train_files)} train, {len(val_files)} val episodes from {path}")
            
            all_files.extend([
                (f, compressed, use_filter, "train") for f in train_files
            ])
            all_files.extend([
                (f, compressed, use_filter, "val") for f in val_files
            ])
        
        # Process all episodes
        train_episodes_data = []
        val_episodes_data = []
        total_filtered = 0
        
        # For statistics computation
        all_states = []
        all_actions = []
        all_images = []
        
        for file_path, compressed, use_filter, split in tqdm(all_files, desc="Processing episodes"):
            episode_sample, num_filtered, ep_states, ep_actions, ep_images = self._parse_episode(
                file_path, compressed, use_filter
            )
            total_filtered += num_filtered
            
            if split == "train":
                train_episodes_data.append(episode_sample)
                # Only use training data for statistics
                if self.compute_stats:
                    all_states.extend(ep_states)
                    all_actions.extend(ep_actions)
                    all_images.extend(ep_images)
            else:
                val_episodes_data.append(episode_sample)
        
        # Compute dataset statistics
        dataset_stats = None
        if self.compute_stats:
            print("Computing dataset statistics for normalization...")
            dataset_stats = compute_dataset_stats(all_states, all_actions, all_images, self.image_key)
            
            print("Dataset statistics:")
            for key, stats in dataset_stats.items():
                print(f"  {key}:")
                print(f"    mean: {stats['mean'].shape} - {stats['mean'].flatten()[:5]}...")
                print(f"    std:  {stats['std'].shape} - {stats['std'].flatten()[:5]}...")
        
        # Create final dataset
        dataset = {
            "train": {
                "episodes": train_episodes_data,
                "num_episodes": len(train_episodes_data),
                "total_steps": sum(ep["episode_metadata"]["episode_length"] for ep in train_episodes_data)
            },
            "val": {
                "episodes": val_episodes_data,
                "num_episodes": len(val_episodes_data),
                "total_steps": sum(ep["episode_metadata"]["episode_length"] for ep in val_episodes_data)
            },
            "metadata": {
                "image_key": self.image_key,
                "total_filtered_actions": total_filtered,
                "data_configs": self.data_configs,
                "dataset_stats": dataset_stats,  # Pi-zero needs this!
            }
        }
        
        # Save dataset
        torch.save(dataset, output_file)
        print(f"Saved dataset to {output_file}")
        print(f"Train: {dataset['train']['num_episodes']} episodes, {dataset['train']['total_steps']} steps")
        print(f"Val: {dataset['val']['num_episodes']} episodes, {dataset['val']['total_steps']} steps")
        print(f"Total filtered actions: {total_filtered}")
        
        return dataset


def create_pi_zero_rlds_dataset(
    data_paths: List[str],
    output_file: str = "pi_zero_rlds_dataset.pt",
    image_key: str = "observation.images.image",
    train_episodes: int = 70,
    val_episodes: int = 5,
    spare_episodes: int = 5,
    apply_filter: bool = True,
    compute_stats: bool = True
):
    """Main function to create the dataset with configurable parameters."""
    
    # Configure your data sources
    data_configs = []
    for path in data_paths:
        data_configs.append({
            "path": path,
            "compressed": True,  # Assume .npz files with arr_0
            "filter": apply_filter,
        })
    
    # Create dataset builder
    builder = PiZeroDatasetBuilder(
        data_configs=data_configs,
        image_key=image_key,
        apply_action_filter=apply_filter,
        success_threshold=6,
        compute_stats=compute_stats
    )
    
    # Build and save dataset
    dataset = builder.build_dataset(
        output_file=output_file,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        spare_episodes=spare_episodes
    )
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Create pi-zero compatible dataset from episode files")
    parser.add_argument("--data_paths", nargs="+", required=True,
                        help="Paths to directories containing episode .npz files")
    parser.add_argument("--output_file", default="pi_zero_rlds_dataset.pt",
                        help="Output dataset file path")
    parser.add_argument("--image_key", default="observation.images.image",
                        help="Key name for images (must match pi-zero config)")
    parser.add_argument("--train_episodes", type=int, default=70,
                        help="Number of episodes for training")
    parser.add_argument("--val_episodes", type=int, default=5,
                        help="Number of episodes for validation")
    parser.add_argument("--spare_episodes", type=int, default=5,
                        help="Number of episodes to hold out")
    parser.add_argument("--no_filter", action="store_true",
                        help="Disable action filtering")
    parser.add_argument("--no_stats", action="store_true",
                        help="Skip computing dataset statistics")
    
    args = parser.parse_args()
    
    create_pi_zero_rlds_dataset(
        data_paths=args.data_paths,
        output_file=args.output_file,
        image_key=args.image_key,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        spare_episodes=args.spare_episodes,
        apply_filter=not args.no_filter,
        compute_stats=not args.no_stats
    )


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # Default behavior for backward compatibility
        create_pi_zero_rlds_dataset(
            data_paths=["/nvme_data/tonghe/RL4VLA/ManiSkill/mp_collect/PutOnPlateInScene25Single-v1/75/data"],
            output_file="pi_zero_rlds_dataset.pt"
        )
    else:
        main() 