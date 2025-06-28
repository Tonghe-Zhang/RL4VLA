import os
import glob
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm



### this code only works for the dataset where there is only one type of image in the observation. 
# we just rename it. 
# to expand to multiple types of images, this code must be modified. 


def create_pi_zero_dataset(data_dir, output_file, image_key):
    """
    Aggregates all episode .npz files in data_dir into a single .pt file for pi-zero supervised finetuning.
    Args:
        data_dir (str): Directory containing .npz files.
        output_file (str): Output .pt file path.
        image_key (str): Name of image key to be renamed. this should align with your pi-zero configuration input_features
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {data_dir}")

    states = []
    actions = []
    images = []
    tasks = []
    to_tensor = transforms.ToTensor()

    for f in tqdm(npz_files, desc="Processing episodes"):
        with np.load(f, allow_pickle=True) as data:
            content = data['arr_0'].item()
            states.append(torch.from_numpy(content['state']))
            actions.append(torch.from_numpy(content['action']))
            img = to_tensor(content['image'])
            images.append(img)
            tasks.append(str(content['instruction'][0]))
    
    # save the dataset to a file
    dataset = {
        "observation.state": torch.stack(states, dim=0),
        "action": torch.stack(actions, dim=0),
        "task": tasks,
    }
    dataset[image_key] = torch.stack(images, dim=0)

    torch.save(dataset, output_file)
    print(f"Saved dataset to {output_file}")

if __name__ == "__main__":
    # Example usage
    data_dir = 'ManiSkill/mp_collect/PutOnPlateInScene25Single-v1/75/data'
    output_file = 'pi_zero_dataset.pt'
    image_key = 'observation.images.image'  # Change as needed
    create_pi_zero_dataset(data_dir, output_file, image_key)