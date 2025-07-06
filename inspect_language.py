from torchvision import transforms
import numpy as np
import os
from pathlib import Path
dir_path='ManiSkill/mp_collect/PutOnPlateInScene25Main-v3/16400/data'


# Get all .npz files in the directory
data_dir = Path(dir_path)
npz_files = list(data_dir.glob('*.npz'))

print(f"Found {len(npz_files)} .npz files in {dir_path}")
print("=" * 80)

# Process each .npz file and print its language instruction
for i, file_path in enumerate(npz_files):
    print(f"File {i+1}/{len(npz_files)}: {file_path.name}")
    
    try:
        with np.load(file_path, allow_pickle=True) as data:
            real_content = data['arr_0'].item()
            
            # Extract and print the instruction
            instruction = real_content['instruction']
            if isinstance(instruction, (list, tuple)) and len(instruction) > 0:
                print(f"  Instruction: {instruction[0]}")
            else:
                print(f"  Instruction: {instruction}")
                
    except Exception as e:
        print(f"  Error loading file: {e}")
    
    print("-" * 40)

print("Done processing all files!") 