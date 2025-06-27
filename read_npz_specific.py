import numpy as np
file_path='ManiSkill/mp_collect/PutOnPlateInScene25Single-v1/75/data/success_proc_0_numid_67_epsid_67.npz'
with np.load(file_path, allow_pickle=True) as data:
    real_content=data['arr_0'].item()
    print(f"real_content={type(real_content)}")
    print(f"real_content={real_content.keys()}")
    print(f"real_content['state']={type(real_content['state'])}, {real_content['state'].shape}")
    print(f"real_content['action']={type(real_content['action'])}, {real_content['action'].shape}")
    print(f"real_content['image']={type(real_content['image'])}, {len(real_content['image'])}, each item is {type(real_content['image'][0])}")
    print(f"real_content['instruction']={type(real_content['instruction'])}, {len(real_content['instruction'])}, content={real_content['instruction'][0]}")