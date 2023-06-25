# [ICLR 2023] CFlowNets: Continuous Control with Generative Flow Networks

[![arXiv](https://img.shields.io/badge/arXiv-2303.02430-b31b1b.svg)](https://arxiv.org/abs/2303.02430)

Official codebase for paper [CFlowNets: Continuous Control with Generative Flow Networks](https://arxiv.org/abs/2303.02430).


## Prerequisites

#### Install dependencies

See `requirments.txt` file for more information about how to install the dependencies.

#### Install mujoco 210
```bash
cd ~/download/
wget -c https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
cd ~/.mujoco
tar -zxvf ~/download/mujoco210-linux-x86_64.tar.gz mujoco210

tar -zxvf ~/download/mujoco210-linux-x86_64.tar.gz mujoco210


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
