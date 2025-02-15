# StoryWeaver
This is an official implementation of AAAI 2025 paper [StoryWeaver: A Unified World Model for Knowledge-Enhanced Story Character Customization](https://arxiv.org/abs/2412.07375). The proposed StoryWeaver can achieve both single- and multi-character based story visualization within a unified model. For more detailed explanations and supplementary materials, please refer to our [arXiv](https://arxiv.org/abs/2412.07375) version. At the same time, we will continuously update the arXiv paper with more experiments.

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
