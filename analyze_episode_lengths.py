#!/usr/bin/env python3
"""
Standalone script to analyze episode lengths from a pi-zero SFT dataset.

Usage:
    python analyze_episode_lengths.py /path/to/dataset.pt
    python analyze_episode_lengths.py /nvme_data/tonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/1000/pi0_sft.pt
"""

import sys
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Any

# Import the PiZeroSFTDataset class to enable loading
from episode2dataset_sft_style import PiZeroSFTDataset


def calculate_episode_length_stats(dataset_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Calculate episode length statistics from a dataset created by PiZeroSFTDatasetBuilder.
    
    Args:
        dataset_path: Path to the .pt dataset file
        verbose: Whether to print detailed statistics
    
    Returns:
        Dictionary containing episode length statistics:
        {
            "train": {
                "min": int,
                "max": int, 
                "mean": float,
                "std": float,
                "median": float,
                "total_episodes": int,
                "total_samples": int,
                "episode_lengths": List[int]
            },
            "val": {...},  # same structure as train
            "combined": {...}  # stats for train + val combined
        }
    """
    print(f"Loading dataset from {dataset_path}")
    
    try:
        dataset = torch.load(dataset_path, map_location='cpu')
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {dataset_path}: {e}")
    
    # Validate dataset structure
    if not isinstance(dataset, dict) or "train" not in dataset or "val" not in dataset:
        raise ValueError("Dataset must contain 'train' and 'val' keys")
    
    def analyze_split(split_name: str, split_data) -> Dict[str, Any]:
        """Analyze episode lengths for a single split (train/val)"""
        if hasattr(split_data, 'samples'):
            # PiZeroSFTDataset object
            samples = split_data.samples
        elif isinstance(split_data, list):
            # Direct list of samples
            samples = split_data
        else:
            raise ValueError(f"Unexpected split_data type: {type(split_data)}")
        
        if not samples:
            return {
                "min": 0, "max": 0, "mean": 0.0, "std": 0.0, "median": 0.0,
                "total_episodes": 0, "total_samples": 0, "episode_lengths": []
            }
        
        # Group samples by episode_index
        episodes = defaultdict(list)
        for sample in samples:
            episode_idx = sample["episode_index"]
            episodes[episode_idx].append(sample)
        
        # Calculate length of each episode
        episode_lengths = []
        for episode_idx, episode_samples in episodes.items():
            episode_length = len(episode_samples)
            episode_lengths.append(episode_length)
        
        episode_lengths = sorted(episode_lengths)
        
        if not episode_lengths:
            return {
                "min": 0, "max": 0, "mean": 0.0, "std": 0.0, "median": 0.0,
                "total_episodes": 0, "total_samples": 0, "episode_lengths": []
            }
        
        # Calculate statistics
        min_length = min(episode_lengths)
        max_length = max(episode_lengths)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        median_length = np.median(episode_lengths)
        
        return {
            "min": min_length,
            "max": max_length,
            "mean": float(mean_length),
            "std": float(std_length),
            "median": float(median_length),
            "total_episodes": len(episode_lengths),
            "total_samples": len(samples),
            "episode_lengths": episode_lengths
        }
    
    # Analyze train and val splits
    train_stats = analyze_split("train", dataset["train"])
    val_stats = analyze_split("val", dataset["val"])
    
    # Calculate combined statistics
    combined_lengths = train_stats["episode_lengths"] + val_stats["episode_lengths"]
    if combined_lengths:
        combined_stats = {
            "min": min(combined_lengths),
            "max": max(combined_lengths),
            "mean": float(np.mean(combined_lengths)),
            "std": float(np.std(combined_lengths)),
            "median": float(np.median(combined_lengths)),
            "total_episodes": len(combined_lengths),
            "total_samples": train_stats["total_samples"] + val_stats["total_samples"],
            "episode_lengths": sorted(combined_lengths)
        }
    else:
        combined_stats = {
            "min": 0, "max": 0, "mean": 0.0, "std": 0.0, "median": 0.0,
            "total_episodes": 0, "total_samples": 0, "episode_lengths": []
        }
    
    results = {
        "train": train_stats,
        "val": val_stats,
        "combined": combined_stats
    }
    
    if verbose:
        print("\n" + "="*60)
        print("EPISODE LENGTH STATISTICS")
        print("="*60)
        
        for split_name, stats in results.items():
            if split_name == "combined":
                print(f"\n{split_name.upper()} (Train + Val):")
            else:
                print(f"\n{split_name.upper()}:")
            print(f"  Total Episodes: {stats['total_episodes']}")
            print(f"  Total Samples:  {stats['total_samples']}")
            print(f"  Min Length:     {stats['min']}")
            print(f"  Max Length:     {stats['max']}")
            print(f"  Mean Length:    {stats['mean']:.2f}")
            print(f"  Std Length:     {stats['std']:.2f}")
            print(f"  Median Length:  {stats['median']:.2f}")
            
            # Show distribution
            if stats['episode_lengths']:
                lengths = stats['episode_lengths']
                print(f"  Length Distribution:")
                print(f"    25th percentile: {np.percentile(lengths, 25):.1f}")
                print(f"    75th percentile: {np.percentile(lengths, 75):.1f}")
                print(f"    95th percentile: {np.percentile(lengths, 95):.1f}")
                
                # Show first few and last few lengths
                if len(lengths) > 10:
                    print(f"  First 5 lengths: {lengths[:5]}")
                    print(f"  Last 5 lengths:  {lengths[-5:]}")
                else:
                    print(f"  All lengths: {lengths}")
        
        # Recommendations for multi-step training
        print("\n" + "="*60)
        print("RECOMMENDATIONS FOR MULTI-STEP TRAINING")
        print("="*60)
        min_length = combined_stats['min']
        print(f"Maximum safe horizon_steps: {min_length}")
        print(f"Recommended horizon_steps for 100% data usage: {min_length}")
        print(f"Recommended horizon_steps for 95% data usage: {int(np.percentile(combined_stats['episode_lengths'], 5))}")
        print(f"Recommended horizon_steps for 90% data usage: {int(np.percentile(combined_stats['episode_lengths'], 10))}")
        
        # Show how many episodes would be lost with different horizon_steps
        print(f"\nData retention with different horizon_steps:")
        for horizon in [1, 5, 10, 15, 20, 25, 30, 50]:
            valid_episodes = sum(1 for length in combined_stats['episode_lengths'] if length >= horizon)
            retention_pct = (valid_episodes / combined_stats['total_episodes']) * 100 if combined_stats['total_episodes'] > 0 else 0
            print(f"  horizon_steps={horizon:2d}: {valid_episodes:3d}/{combined_stats['total_episodes']:3d} episodes ({retention_pct:.1f}%)")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_episode_lengths.py <dataset_path>")
        print("Example: python analyze_episode_lengths.py /path/to/dataset.pt")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    try:
        stats = calculate_episode_length_stats(dataset_path, verbose=True)
        
        # Print a summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        combined = stats['combined']
        print(f"📊 Dataset Analysis Complete!")
        print(f"📁 Dataset: {dataset_path}")
        print(f"📈 Total Episodes: {combined['total_episodes']}")
        print(f"📋 Total Samples: {combined['total_samples']}")
        print(f"📏 Episode Length Range: {combined['min']}-{combined['max']} (avg: {combined['mean']:.1f})")
        print(f"🎯 Recommended horizon_steps: {combined['min']} (for 100% data usage)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 