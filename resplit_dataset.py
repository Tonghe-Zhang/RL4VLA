#!/usr/bin/env python3

import torch
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Any, List
import json
from episode2dataset_sft_style import PiZeroSFTDataset
def resplit_dataset(dataset_path: str, 
                   train_episodes_requested: int = 16000,
                   val_episodes_requested: int = 384,
                   output_path: str = None,
                   shuffle: bool = True,
                   random_seed: int = 42):
    """
    Re-split an existing dataset to maintain the desired train/validation ratio.
    
    Args:
        dataset_path: Path to the existing dataset .pt file
        train_episodes_requested: Originally requested number of training episodes
        val_episodes_requested: Originally requested number of validation episodes
        output_path: Path to save the re-split dataset (if None, overwrites original)
        shuffle: Whether to shuffle samples before splitting
        random_seed: Random seed for reproducible shuffling
    """
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = torch.load(dataset_path)
    
    # Examine dataset structure
    print("\n=== Dataset Structure ===")
    print(f"Keys: {list(dataset.keys())}")
    
    if 'train' in dataset:
        print(f"Train samples: {len(dataset['train'])}")
    if 'val' in dataset:
        print(f"Val samples: {len(dataset['val'])}")
    
    # Extract metadata
    metadata = dataset.get('metadata', {})
    print(f"Metadata keys: {list(metadata.keys())}")
    
    # Get all training samples (since they're currently all in train)
    train_samples = dataset['train'].samples if hasattr(dataset['train'], 'samples') else []
    val_samples = dataset['val'].samples if hasattr(dataset['val'], 'samples') else []
    
    print(f"\nCurrent samples: {len(train_samples)} train, {len(val_samples)} val")
    
    # Combine all samples
    all_samples = train_samples + val_samples
    total_samples = len(all_samples)
    
    print(f"Total samples to re-split: {total_samples}")
    
    # Calculate the desired ratio
    total_requested = train_episodes_requested + val_episodes_requested
    train_ratio = train_episodes_requested / total_requested
    val_ratio = val_episodes_requested / total_requested
    
    print(f"\nDesired ratio: {train_ratio:.1%} train, {val_ratio:.1%} val")
    
    # Calculate new split sizes
    new_train_size = int(total_samples * train_ratio)
    new_val_size = total_samples - new_train_size
    
    print(f"New split: {new_train_size} train, {new_val_size} val")
    
    # Shuffle if requested
    if shuffle:
        print(f"Shuffling samples with seed {random_seed}")
        np.random.seed(random_seed)
        indices = np.random.permutation(total_samples)
        all_samples = [all_samples[i] for i in indices]
    
    # Split samples
    new_train_samples = all_samples[:new_train_size]
    new_val_samples = all_samples[new_train_size:new_train_size + new_val_size]
    
    print(f"Final split: {len(new_train_samples)} train, {len(new_val_samples)} val")
    
    # Import the dataset class
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    
    # Create new dataset objects
    new_train_dataset = PiZeroSFTDataset(new_train_samples)
    new_val_dataset = PiZeroSFTDataset(new_val_samples)
    
    # Update metadata
    new_metadata = metadata.copy()
    new_metadata['train_samples'] = len(new_train_samples)
    new_metadata['val_samples'] = len(new_val_samples)
    new_metadata['resplit_info'] = {
        'original_train_samples': len(train_samples),
        'original_val_samples': len(val_samples),
        'requested_train_episodes': train_episodes_requested,
        'requested_val_episodes': val_episodes_requested,
        'shuffled': shuffle,
        'random_seed': random_seed if shuffle else None,
    }
    
    # Create new dataset
    new_dataset = {
        'train': new_train_dataset,
        'val': new_val_dataset,
        'metadata': new_metadata
    }
    
    # Save the re-split dataset
    if output_path is None:
        output_path = dataset_path
    
    output_path = Path(output_path)
    
    # # Create backup of original if overwriting
    # if output_path == Path(dataset_path):
    #     backup_path = output_path.with_suffix('.backup.pt')
    #     print(f"Creating backup: {backup_path}")
    #     torch.save(dataset, backup_path)
    
    print(f"Saving re-split dataset to: {output_path}")
    torch.save(new_dataset, output_path)
    
    # Save summary
    summary_path = output_path.with_suffix('.resplit_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Dataset Re-split Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Original dataset: {dataset_path}\n")
        f.write(f"Output dataset: {output_path}\n")
        f.write(f"Requested ratio: {train_episodes_requested}:{val_episodes_requested}\n")
        f.write(f"Actual ratio: {train_ratio:.1%}:{val_ratio:.1%}\n")
        f.write(f"\nOriginal split:\n")
        f.write(f"  Train: {len(train_samples)} samples\n")
        f.write(f"  Val: {len(val_samples)} samples\n")
        f.write(f"\nNew split:\n")
        f.write(f"  Train: {len(new_train_samples)} samples\n")
        f.write(f"  Val: {len(new_val_samples)} samples\n")
        f.write(f"\nSettings:\n")
        f.write(f"  Shuffled: {shuffle}\n")
        f.write(f"  Random seed: {random_seed if shuffle else 'N/A'}\n")
    
    print(f"Summary saved to: {summary_path}")
    
    return new_dataset

def main():
    parser = argparse.ArgumentParser(description="Re-split existing dataset to maintain train/validation ratio")
    parser.add_argument("--dataset_path", required=True, help="Path to existing dataset .pt file")
    parser.add_argument("--train_episodes", type=int, default=12000, help="Originally requested training episodes")
    parser.add_argument("--val_episodes", type=int, default=800, help="Originally requested validation episodes")
    parser.add_argument("--output_path", help="Output path (if None, overwrites original)")
    parser.add_argument("--no_shuffle", action="store_true", help="Don't shuffle samples before splitting")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for shuffling")
    
    args = parser.parse_args()
    
    resplit_dataset(
        dataset_path=args.dataset_path,
        train_episodes_requested=args.train_episodes,
        val_episodes_requested=args.val_episodes,
        output_path=args.output_path,
        shuffle=not args.no_shuffle,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main() 
    
    
    """
    
    python resplit_dataset.py --dataset_path="./datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/pi0_sft.pt" \
        --train_episodes=12000 --val_episodes=800
    
    """