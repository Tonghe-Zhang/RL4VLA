from torchvision import transforms
import numpy as np
file_path='SimplerEnv/octo_collect/PutCarrotOnPlateInScene-v1/75/data/data_0000.npy'
data=np.load(file_path, allow_pickle=True)
real_content=data.item()

img_ts=transforms.ToTensor()(real_content['image'][0])
# arr.keys()==dict_keys(['image', 'instruction', 'action', 'info'])
# arr['action'] is a list of length 80 and each element is a list of length 7 (6 dofs eef + 1 gripper 1/0 status)

# arr['image'] is a list of length 81 and each element is a <class 'PIL.Image.Image'>. you can convert them to 
# img_ts=transforms.ToTensor()(real_content['image'][0])
# arr['instruction'] is a single string 'put carrot on plate'
print(f"img_ts={img_ts.shape}") # [3,480,640] image