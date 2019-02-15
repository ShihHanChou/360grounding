# Self-view Grounding Given a Narrated 360° Video

[Shih-Han Chou](https://shihhanchou.github.io/), [Yi-Chun Chen](https://chenyichun.github.io/), [Kuo-Hao Zeng](https://kuohaozeng.github.io/), [Hou-Ning Hu](https://eborboihuc.github.io/), [Jianlong Fu](https://www.microsoft.com/en-us/research/people/jianf/), [Min Sun](http://aliensunmin.github.io/)

Association for the Advancement of Artificial Intelligence (AAAI) ,2018  
Official Implementation of AAAI 2018 paper "Self-view Grounding Given a Narrated 360° Video" in Pytorch.

Project page: [http://aliensunmin.github.io/project/360grounding/](http://aliensunmin.github.io/project/360grounding/)  
Paper: [ArXiv](https://arxiv.org/abs/1711.08664), [AAAI18](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16724)

# Prerequisites
* Linux  
* NVIDIA GPU + CUDA 7.0 + CuDNNv5.1  
* Python 2.7 with numpy  
* Pytorch 0.3.0  

# Getting Started
* Clone this repo
```
git clone https://github.com/ShihHanChou/360grounding.git
```
* Download our [dataset](https://github.com/ShihHanChou/360grounding/blob/master/README.md#dataset)

# Dataset
Please download the dataset [here](https://goo.gl/forms/9DRj4jvWDyRCIsxh2) and place it under `./data`.

# Usage
* To train a model with downloaded dataset:
```
python main.py --batch_size $batch_size$ --epoches $#epoches$ --save_dir $save_directory$ --mode train --video_len $video_sample_length$ --MAX_LENGTH 33
```
* To test a model with downloaded dataset:
```
python main.py --batch_size $batch_size$ --epoches $which_epoches$ --save_dir $save_directory$ --mode test --MAX_LENGTH 30
```

# Cite
If you find our code useful for your research, please cite
```
@inproceedings{chou2018self,
  title={Self-view grounding given a narrated 360 video},
  author={Chou, Shih-Han and Chen, Yi-Chun and Zeng, Kuo-Hao and Hu, Hou-Ning and Fu, Jianlong and Sun, Min},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
```
