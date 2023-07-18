# [ICLR 2023] CFlowNets: Continuous Control with Generative Flow Networks

[![arXiv](https://img.shields.io/badge/arXiv-2303.02430-b31b1b.svg)](https://arxiv.org/abs/2303.02430)

Official codebase for paper [CFlowNets: Continuous Control with Generative Flow Networks](https://arxiv.org/abs/2303.02430).


## Prerequisites

#### Install dependencies

- torch==1.8.1
- gym==0.18.3
- python==3.8.10
- tensorboard==2.5.0

#### Install mujoco 210
```bash
cd ~
wget -c https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
cd ~/.mujoco
tar -zxvf ~/mujoco210-linux-x86_64.tar.gz mujoco210

echo "export LD_LIBRARY_PATH=\$HOME/.mujoco/mujoco210/bin:\$LD_LIBRARY_PATH" >> ~/.profile
echo "export MUJOCO_PY_MUJOCO_PATH=\"\$HOME/.mujoco/mujoco210\"" >> ~/.profile
echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:/usr/lib/nvidia\"" >> ~/.profile
pip install -U 'mujoco-py<2.2,>=2.1'
```


## Usage

Please follow the instructions below to replicate the results in the paper.

#### Train retrieval network of different environments.
```bash
# Reacher
python Retrieval_Reacher.py
```


#### Train continuous flow network of different environments.
```bash
# Reacher
python CFN_Reacher.py
```

## Remark
To reproduce the results of Figure 3 (a)-(c) in the paper, it is necessary to set the is_max variable in the select_action function to 0 to strictly sample more diverse results based on their probabilities according to the flow network. If you want to maximize the reward and reproduce the results in Figure 3 (d)-(f) in the paper, you need to set the is_max variable to 1 in order to maximize the reward.

The way of sampling according to the flow network is diverse, and as mentioned in Remark 1 in the paper:

Remark 1. After the training process, for tasks that require a larger reward, we can sample actions with the maximum flow output in P during the test process to obtain a relatively higher reward. The output of the flow model is used is flexible, and we can adjust it for different tasks.


## Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{li2023,
  title={CFlowNets: Continuous Control with Generative Flow Networks},
  author={Yinchuan Li and Shuang Luo and Haozhi Wang and Jianye Hao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## Contact

Please feel free to contact me via email (<luoshuang@zju.edu.cn>) if you are interested in my research :)
