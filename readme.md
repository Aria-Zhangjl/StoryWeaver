# StoryWeaver
This is an official implementation of AAAI 2025 paper [StoryWeaver: A Unified World Model for Knowledge-Enhanced Story Character Customization](https://ojs.aaai.org/index.php/AAAI/article/view/33079). The proposed StoryWeaver can achieve both single- and multi-character based story visualization within a unified model. For more detailed explanations and supplementary materials, please refer to our [arXiv](https://arxiv.org/abs/2412.07375) version. At the same time, we will continuously update the arXiv paper with more experiments.

## 🚀 Overview
![](visualization/whole.png)

## 🚩 Updates/Todo List

- [ ] Adding more experiments results on more stories world, e.g., Flintstons, Pokemon,...
- [x] Code Released.
- [x] Paper Released.
## 🖥️  Getting Start
### Environment
The codebase is tested on 
* Python 3.8

For additional python libraries, please install by:

```
pip install -r requirements.txt
```
### Training StoryWeaver

use the shell script,

```
bash train.sh
```

### Sample from StoryWeaver

use the shell script,

```
bash sample.sh
```

> [!TIP]
> We highly recommend performing Knowledge-Enhanced Spatial Guidance on the coarse class label too for each character in the text descriptions. For example, the coarse label for character Loopy is beaver. Please refer to the [code](https://github.com/Aria-Zhangjl/StoryWeaver/blob/main/code/TBC_dataset.py) of the dataset for more details.

### Visual Results

#### Single-Character Story Visualization
![](visualization/single.png)

#### Multi-Character Story Visualization
![](visualization/multi.png)


## 🖊️ Citation
If StoryWeaver is helpful for your research or you wish to refer the baseline results published here, we'd really appreciate it if you could cite this paper:
```
@article{Zhang_Tang_Zhang_Lv_Sun_2025,
title={StoryWeaver: A Unified World Model for Knowledge-Enhanced Story Character Customization},
volume={39}, url={https://ojs.aaai.org/index.php/AAAI/article/view/33079}, DOI={10.1609/aaai.v39i9.33079},
number={9}, journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Zhang, Jinlu and Tang, Jiji and Zhang, Rongsheng and Lv, Tangjie and Sun, Xiaoshuai},
year={2025}, month={Apr.}, pages={9951-9959} }
