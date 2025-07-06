
# Generate raw data episodes

## PutOnPlateInScene25Main-v3
* Task: Pick-and-place
* Env id: PutOnPlateInScene25Main-v3
* Domain randomization: Yes
* Data source: Motion Planning
* Number of trajectories: 16,384
* Robot type: widowx
* Command:
```bash
conda activate rlvla_env
cd ManiSkill
cuda=2
CUDA_VISIBLE_DEVICES=$cuda \
python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
  -e "PutOnPlateInScene25Main-v3" \
  --save_video --save_data --num_procs 16 --num_traj 16384 --seed=100
```


# Merge and normalize raw episodes into SFT dataset 

## PutOnPlateInScene25Main-v3
* Task: Pick-and-place
* Env id: PutOnPlateInScene25Main-v3
* Domain randomization: Yes
* Data source: Motion Planning
* Number of trajectories: 16,384
* Robot type: widowx
* Train/val split: 2.4% validation
* Command:
```bash
conda activate rlvla_env
cuda=2
CUDA_VISIBLE_DEVICES=$cuda \
python episode2dataset_sft_style.py \
--data_paths="/nvme_data/tonghe/RL4VLA/ManiSkill/mp_collect/PutOnPlateInScene25Main-v3/12800/data" \
--task_name="PutOnPlateInScene25Main-v3" \
--output_dir="./datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/12800" \
--dataset_name="pi0_sft" \
--train_episodes=12800 \
--val_episodes=0
```

```bash
# in fact we only generated 12800 episodes. so 
CUDA_VISIBLE_DEVICES=$cuda python episode2dataset_sft_style.py --data_paths="/nvme_data/tonghe/RL4VLA/ManiSkill/mp_collect/PutOnPlateInScene25Main-v3/16384/data" --task_name="PutOnPlateInScene25Main-v3" --output_dir="./datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3" --dataset_name="pi0_sft" --train_episodes=16000 --val_episodes=
```


```bash
# build a smaller dataset to save space, which only contains the first s1,000 episodes. 
conda activate rlvla_env
CUDA_VISIBLE_DEVICES='3,4,5' \
python episode2dataset_sft_style.py \
--data_paths="/nvme_data/tonghe/RL4VLA/ManiSkill/mp_collect/PutOnPlateInScene25Main-v3/12800/data" \
--task_name="PutOnPlateInScene25Main-v3" \
--output_dir="./datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/1000" \
--dataset_name="pi0_sft" \
--train_episodes=900 \
--val_episodes=100
```


  