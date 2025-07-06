from torchvision import transforms
import numpy as np
file_paths=[
    '/nvme_data/tonghe/RL4VLA/ManiSkill/mp_collect/PutOnPlateInScene25Single-v1/75/data/success_proc_0_numid_67_epsid_67.npz',
    'ManiSkill/mp_collect/PutOnPlateInScene25Main-v3/16384/data/success_proc_0_numid_0_epsid_0.npz'
]
file_path=file_paths[1]

dir_path='ManiSkill/mp_collect/PutOnPlateInScene25Main-v3/16400/data'
with np.load(file_path, allow_pickle=True) as data:
    real_content=data['arr_0'].item()
    img_ts=transforms.ToTensor()(real_content['image'][0])
    print(f"real_content={type(real_content)}")
    print(f"real_content={real_content.keys()}")
    print(f"real_content['state']={type(real_content['state'])}, {real_content['state'].shape}")
    print(f"real_content['action']={type(real_content['action'])}, {real_content['action'].shape}")
    print(f"real_content['image']={type(real_content['image'])}, {len(real_content['image'])}, each item is {type(real_content['image'][0])}, convert one image to tensor: {type(img_ts)}, {img_ts.shape}, {img_ts.dtype}, range: [{img_ts.min().item()}, {img_ts.max().item()}]")
    print(f"real_content['is_image_encode']={real_content['is_image_encode']}")
    print(f"real_content['instruction']={type(real_content['instruction'])}, len={len(real_content['instruction'])}, first element={type(real_content['instruction'][0])}: {real_content['instruction'][0]}") 