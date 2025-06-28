import os
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


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


# output_dir/
# ├── pi_zero_dataset.pt                    # Main dataset file
# ├── pi_zero_dataset_normalization.pt      # Normalization stats (PyTorch)
# ├── pi_zero_dataset_normalization.json    # Normalization stats (JSON)
# ├── pi_zero_dataset_normalization_summary.txt  # Stats summary
# └── pi_zero_dataset_summary.txt           # Dataset summary

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


def compute_dataset_stats(all_states, all_actions, all_images, image_key, normalization_mapping):
    """
    Compute dataset statistics for pi-zero normalization based on normalization_mapping.
    
    CRITICAL: This must exactly match pi-zero's expected format!
    - Keys must match feature keys exactly
    - Values must be torch.Tensor (float32)
    - Shapes must match pi-zero's expectations
    
    Args:
        all_states: List of state tensors [state_dim]
        all_actions: List of action tensors [action_dim]
        all_images: List of image tensors [3, H, W]
        image_key: Key name for images (e.g. "observation.images.image")
        normalization_mapping: Dict mapping modality types to normalization modes
    
    Returns:
        Dict with statistics exactly as pi-zero expects:
        - MEAN_STD: {"mean": torch.Tensor, "std": torch.Tensor}
        - MIN_MAX: {"min": torch.Tensor, "max": torch.Tensor}
        - IDENTITY: {} (no stats needed)
    """
    stats = {}
    
    # State statistics - compute across ALL timesteps from ALL episodes
    if all_states and normalization_mapping.get("STATE") != NormalizationMode.IDENTITY:
        # Stack all state tensors: [total_timesteps, state_dim]
        states_tensor = torch.stack(all_states, dim=0)
        
        if normalization_mapping.get("STATE") == NormalizationMode.MEAN_STD:
            # Pi-zero expects: [state_dim] tensors
            stats["observation.state"] = {
                "mean": states_tensor.mean(dim=0).to(dtype=torch.float32),  # [state_dim]
                "std": states_tensor.std(dim=0).to(dtype=torch.float32),    # [state_dim]
            }
        elif normalization_mapping.get("STATE") == NormalizationMode.MIN_MAX:
            stats["observation.state"] = {
                "min": states_tensor.min(dim=0)[0].to(dtype=torch.float32),  # [state_dim]
                "max": states_tensor.max(dim=0)[0].to(dtype=torch.float32),  # [state_dim]
            }
    
    # Action statistics - compute across ALL timesteps from ALL episodes
    if all_actions and normalization_mapping.get("ACTION") != NormalizationMode.IDENTITY:
        # Stack all action tensors: [total_timesteps, action_dim]
        actions_tensor = torch.stack(all_actions, dim=0)
        
        if normalization_mapping.get("ACTION") == NormalizationMode.MEAN_STD:
            # Pi-zero expects: [action_dim] tensors
            stats["action"] = {
                "mean": actions_tensor.mean(dim=0).to(dtype=torch.float32),  # [action_dim]
                "std": actions_tensor.std(dim=0).to(dtype=torch.float32),    # [action_dim]
            }
        elif normalization_mapping.get("ACTION") == NormalizationMode.MIN_MAX:
            stats["action"] = {
                "min": actions_tensor.min(dim=0)[0].to(dtype=torch.float32),  # [action_dim]
                "max": actions_tensor.max(dim=0)[0].to(dtype=torch.float32),  # [action_dim]
            }
    
    # Image statistics - compute across ALL timesteps from ALL episodes (per-channel)
    if all_images and normalization_mapping.get("VISUAL") != NormalizationMode.IDENTITY:
        # Stack all image tensors: [total_timesteps, 3, H, W]
        images_tensor = torch.stack(all_images, dim=0)
        
        if normalization_mapping.get("VISUAL") == NormalizationMode.MEAN_STD:
            # Pi-zero expects: [3, 1, 1] tensors for images (per-channel stats)
            stats[image_key] = {
                "mean": images_tensor.mean(dim=(0, 2, 3), keepdim=True).to(dtype=torch.float32),  # [3, 1, 1]
                "std": images_tensor.std(dim=(0, 2, 3), keepdim=True).to(dtype=torch.float32),    # [3, 1, 1]
            }
        elif normalization_mapping.get("VISUAL") == NormalizationMode.MIN_MAX:
            # Compute per-channel min/max: [3, 1, 1]
            # First get per-channel min/max across spatial dimensions
            channel_mins = images_tensor.min(dim=0)[0]  # [3, H, W]
            channel_maxs = images_tensor.max(dim=0)[0]  # [3, H, W]
            # Then get global min/max per channel
            global_mins = channel_mins.view(3, -1).min(dim=1)[0].view(3, 1, 1)  # [3, 1, 1]
            global_maxs = channel_maxs.view(3, -1).max(dim=1)[0].view(3, 1, 1)  # [3, 1, 1]
            
            stats[image_key] = {
                "min": global_mins.to(dtype=torch.float32),  # [3, 1, 1]
                "max": global_maxs.to(dtype=torch.float32),  # [3, 1, 1]
            }
    
    # Validate that all tensors are float32 and have correct shapes
    for key, stat_dict in stats.items():
        for stat_name, tensor_val in stat_dict.items():
            if not isinstance(tensor_val, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor for {key}.{stat_name}, got {type(tensor_val)}")
            if tensor_val.dtype != torch.float32:
                print(f"Warning: Converting {key}.{stat_name} from {tensor_val.dtype} to float32")
                stats[key][stat_name] = tensor_val.to(dtype=torch.float32)
    
    return stats


def validate_pi_zero_stats_compatibility(dataset_stats, normalization_mapping, image_key):
    """
    Validate that dataset_stats are compatible with pi-zero's normalization requirements.
    
    This function checks:
    1. All values are torch.Tensor with dtype=float32
    2. Tensor shapes match pi-zero expectations
    3. Keys match the expected feature names
    4. Statistics match the normalization modes
    
    Args:
        dataset_stats: Computed dataset statistics
        normalization_mapping: Normalization mapping used
        image_key: Image key name
    
    Raises:
        ValueError: If any compatibility issue is found
    """
    print("🔍 Validating pi-zero compatibility...")
    
    expected_keys = set()
    if normalization_mapping.get("STATE") != NormalizationMode.IDENTITY:
        expected_keys.add("observation.state")
    if normalization_mapping.get("ACTION") != NormalizationMode.IDENTITY:
        expected_keys.add("action")
    if normalization_mapping.get("VISUAL") != NormalizationMode.IDENTITY:
        expected_keys.add(image_key)
    
    # Check that we have the expected keys
    actual_keys = set(dataset_stats.keys())
    if actual_keys != expected_keys:
        raise ValueError(f"Key mismatch! Expected: {expected_keys}, Got: {actual_keys}")
    
    for key, stats in dataset_stats.items():
        print(f"  Checking {key}...")
        
        # Determine expected normalization mode
        if key == "observation.state":
            norm_mode = normalization_mapping.get("STATE")
        elif key == "action":
            norm_mode = normalization_mapping.get("ACTION")
        elif key == image_key:
            norm_mode = normalization_mapping.get("VISUAL")
        else:
            raise ValueError(f"Unexpected key: {key}")
        
        # Check statistics match normalization mode
        if norm_mode == NormalizationMode.MEAN_STD:
            required_stats = {"mean", "std"}
        elif norm_mode == NormalizationMode.MIN_MAX:
            required_stats = {"min", "max"}
        else:
            continue  # IDENTITY mode
        
        actual_stats = set(stats.keys())
        if actual_stats != required_stats:
            raise ValueError(f"Stat mismatch for {key}! Expected: {required_stats}, Got: {actual_stats}")
        
        # Check tensor properties
        for stat_name, tensor_val in stats.items():
            if not isinstance(tensor_val, torch.Tensor):
                raise ValueError(f"{key}.{stat_name} must be torch.Tensor, got {type(tensor_val)}")
            
            if tensor_val.dtype != torch.float32:
                raise ValueError(f"{key}.{stat_name} must be float32, got {tensor_val.dtype}")
            
            # Check shapes
            if key == image_key:
                expected_shape = (3, 1, 1)
                if tensor_val.shape != expected_shape:
                    raise ValueError(f"{key}.{stat_name} shape should be {expected_shape}, got {tensor_val.shape}")
            else:
                # State and action should be 1D tensors
                if tensor_val.ndim != 1:
                    raise ValueError(f"{key}.{stat_name} should be 1D tensor, got shape {tensor_val.shape}")
            
            # Check for invalid values
            if torch.isnan(tensor_val).any():
                raise ValueError(f"{key}.{stat_name} contains NaN values")
            if torch.isinf(tensor_val).any():
                raise ValueError(f"{key}.{stat_name} contains infinite values")
            
            print(f"    ✅ {stat_name}: {tensor_val.shape} {tensor_val.dtype}")
    
    print("✅ All checks passed! Dataset stats are pi-zero compatible.")


def save_normalization_stats(stats, output_dir, filename="normalization_stats", normalization_mapping=None):
    """
    Save normalization statistics in multiple formats for convenience.
    
    Args:
        stats: Dictionary containing normalization statistics
        output_dir: Directory to save the files
        filename: Base filename (without extension)
        normalization_mapping: Normalization mapping used
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate compatibility before saving
    if normalization_mapping:
        # Extract image key from stats (assuming there's only one image key)
        image_keys = [k for k in stats.keys() if k not in ["observation.state", "action"]]
        image_key = image_keys[0] if image_keys else None
        if image_key:
            validate_pi_zero_stats_compatibility(stats, normalization_mapping, image_key)
    
    # Save as PyTorch file (for direct loading in pi-zero)
    torch_file = output_dir / f"{filename}.pt"
    torch.save(stats, torch_file)
    print(f"Saved normalization statistics (PyTorch format): {torch_file}")
    
    # Save as JSON for human readability (convert tensors to lists)
    json_stats = {}
    for key, stat_dict in stats.items():
        json_stats[key] = {}
        for stat_name, tensor_val in stat_dict.items():
            json_stats[key][stat_name] = tensor_val.tolist()
    
    json_file = output_dir / f"{filename}.json"
    with open(json_file, 'w') as f:
        json.dump(json_stats, f, indent=2)
    print(f"Saved normalization statistics (JSON format): {json_file}")
    
    # Save summary as text file
    txt_file = output_dir / f"{filename}_summary.txt"
    with open(txt_file, 'w') as f:
        f.write("Dataset Normalization Statistics Summary\n")
        f.write("=" * 50 + "\n\n")
        
        if normalization_mapping:
            f.write("Normalization Mapping:\n")
            for modality, mode in normalization_mapping.items():
                f.write(f"  {modality}: {mode}\n")
            f.write("\n")
        
        f.write("Statistics (Pi-Zero Compatible Format):\n")
        for key, stat_dict in stats.items():
            f.write(f"{key}:\n")
            for stat_name, tensor_val in stat_dict.items():
                f.write(f"  {stat_name}: shape={list(tensor_val.shape)}, dtype={tensor_val.dtype}\n")
                if tensor_val.numel() <= 10:  # Show full tensor if small
                    f.write(f"    values: {tensor_val.flatten().tolist()}\n")
                else:  # Show first few values if large
                    f.write(f"    first 5 values: {tensor_val.flatten()[:5].tolist()}\n")
                    f.write(f"    range: [{tensor_val.min().item():.6f}, {tensor_val.max().item():.6f}]\n")
            f.write("\n")
        
        f.write("Compatibility Check:\n")
        f.write("✅ All tensors are torch.float32\n")
        f.write("✅ All shapes match pi-zero expectations\n")
        f.write("✅ No NaN or infinite values\n")
    
    print(f"Saved normalization statistics summary: {txt_file}")
    
    return torch_file, json_file, txt_file


class PiZeroDatasetBuilder:
    """PyTorch dataset builder for pi-zero in RLDS style (episode-based structure)."""
    
    def __init__(self, 
                 data_configs: List[Dict[str, Any]], 
                 image_key: str = "observation.images.image",
                 apply_action_filter: bool = True,
                 success_threshold: int = 6,
                 compute_stats: bool = True,
                 normalization_mapping: Dict[str, NormalizationMode] = None):
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
            normalization_mapping: Dict mapping modality types to normalization modes
        """
        self.data_configs = data_configs
        self.image_key = image_key
        self.apply_action_filter = apply_action_filter
        self.success_threshold = success_threshold
        self.compute_stats = compute_stats
        self.to_tensor = transforms.ToTensor()
        
        # Default normalization mapping (matches pi-zero defaults)
        if normalization_mapping is None:
            normalization_mapping = {
                "VISUAL": NormalizationMode.IDENTITY,  # Pi-zero handles image normalization internally
                "STATE": NormalizationMode.MEAN_STD,   # Normalize state to mean 0, std 1
                "ACTION": NormalizationMode.MEAN_STD,  # Normalize action to mean 0, std 1
            }
        self.normalization_mapping = normalization_mapping
    
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
                     output_dir: str,
                     dataset_name: str = "pi_zero_dataset", 
                     train_episodes: int = 70, 
                     val_episodes: int = 5,
                     spare_episodes: int = 5) -> Dict[str, Any]:
        """
        Build the complete dataset with train/val splits.
        
        Args:
            output_dir: Directory to save all output files
            dataset_name: Base name for output files (without extension)
            train_episodes: Number of episodes for training
            val_episodes: Number of episodes for validation  
            spare_episodes: Number of episodes to hold out
        """
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
        stats_files = []
        if self.compute_stats:
            print("Computing dataset statistics for normalization...")
            dataset_stats = compute_dataset_stats(
                all_states, all_actions, all_images, self.image_key, self.normalization_mapping
            )
            
            print("Dataset statistics:")
            print(f"Normalization mapping: {self.normalization_mapping}")
            for key, stats in dataset_stats.items():
                print(f"  {key}:")
                for stat_name, stat_val in stats.items():
                    print(f"    {stat_name}: {stat_val.shape} - {stat_val.flatten()[:5]}...")
            
            # Save normalization statistics separately
            stats_files = save_normalization_stats(
                dataset_stats, output_dir, f"{dataset_name}_normalization", self.normalization_mapping
            )
        
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
                "normalization_mapping": {k: v.value for k, v in self.normalization_mapping.items()},
                "normalization_files": [str(f) for f in stats_files] if stats_files else [],
            }
        }
        
        # Save dataset
        dataset_file = output_dir / f"{dataset_name}.pt"
        torch.save(dataset, dataset_file)
        print(f"Saved dataset to {dataset_file}")
        print(f"Train: {dataset['train']['num_episodes']} episodes, {dataset['train']['total_steps']} steps")
        print(f"Val: {dataset['val']['num_episodes']} episodes, {dataset['val']['total_steps']} steps")
        print(f"Total filtered actions: {total_filtered}")
        
        # Create a summary file
        summary_file = output_dir / f"{dataset_name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Pi-Zero Dataset Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Dataset file: {dataset_file.name}\n")
            f.write(f"Image key: {self.image_key}\n")
            f.write(f"Train episodes: {dataset['train']['num_episodes']} ({dataset['train']['total_steps']} steps)\n")
            f.write(f"Val episodes: {dataset['val']['num_episodes']} ({dataset['val']['total_steps']} steps)\n")
            f.write(f"Total filtered actions: {total_filtered}\n")
            f.write(f"Data sources: {len(self.data_configs)}\n")
            for i, config in enumerate(self.data_configs):
                f.write(f"  {i+1}. {config['path']}\n")
            
            f.write(f"\nNormalization mapping:\n")
            for modality, mode in self.normalization_mapping.items():
                f.write(f"  {modality}: {mode}\n")
            
            if stats_files:
                f.write(f"\nNormalization files:\n")
                for stats_file in stats_files:
                    f.write(f"  - {Path(stats_file).name}\n")
        
        print(f"Saved dataset summary: {summary_file}")
        
        return dataset


def create_pi_zero_rlds_dataset(
    data_paths: List[str],
    output_dir: str = "./pi_zero_output",
    dataset_name: str = "pi_zero_dataset",
    image_key: str = "observation.images.image",
    train_episodes: int = 70,
    val_episodes: int = 5,
    spare_episodes: int = 5,
    apply_filter: bool = True,
    compute_stats: bool = True,
    normalization_mapping: Dict[str, NormalizationMode] = None
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
        compute_stats=compute_stats,
        normalization_mapping=normalization_mapping
    )
    
    # Build and save dataset
    dataset = builder.build_dataset(
        output_dir=output_dir,
        dataset_name=dataset_name,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        spare_episodes=spare_episodes
    )
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Create pi-zero compatible dataset from episode files")
    parser.add_argument("--data_paths", nargs="+", required=True,
                        help="Paths to directories containing episode .npz files")
    parser.add_argument("--output_dir", default="./pi_zero_output",
                        help="Output directory for all generated files")
    parser.add_argument("--dataset_name", default="pi_zero_dataset",
                        help="Base name for output files (without extension)")
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
    
    # Normalization arguments
    parser.add_argument("--state_norm", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], 
                        default="MEAN_STD", help="Normalization mode for states")
    parser.add_argument("--action_norm", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], 
                        default="MEAN_STD", help="Normalization mode for actions")
    parser.add_argument("--visual_norm", choices=["MEAN_STD", "MIN_MAX", "IDENTITY"], 
                        default="IDENTITY", help="Normalization mode for images")
    
    args = parser.parse_args()
    
    # Create normalization mapping
    normalization_mapping = {
        "STATE": NormalizationMode(args.state_norm),
        "ACTION": NormalizationMode(args.action_norm),
        "VISUAL": NormalizationMode(args.visual_norm),
    }
    
    create_pi_zero_rlds_dataset(
        data_paths=args.data_paths,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        spare_episodes=args.spare_episodes,
        apply_filter=not args.no_filter,
        compute_stats=not args.no_stats,
        normalization_mapping=normalization_mapping
    )


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # Default behavior for backward compatibility
        create_pi_zero_rlds_dataset(
            data_paths=["/nvme_data/tonghe/RL4VLA/ManiSkill/mp_collect/PutOnPlateInScene25Single-v1/75/data"],
            output_dir="./pi_zero_output",
            dataset_name="pi_zero_dataset"
        )
    else:
        main() 