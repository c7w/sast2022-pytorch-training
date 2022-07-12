# PyTorch 图像分类作业
<center>By c7w</center>

## 作业目标
[本次作业](https://github.com/c7w/sast2022-pytorch-training)是对学习的数据分析技能和人工神经网络知识的入门实践，其主体采用**代码填空**的形式进行。你需要按照本实验指导中的流程，尝试阅读并理解现有的实验框架，然后在现有实验框架的基础上做代码补全，以实现对应子任务所要求的功能。

我们本次作业的最终目标是要利用预先获取到的风景图像数据，训练一个固定图像分辨率的风景图像**多分类**网络。我们会对于给定的一张固定分辨率的图片，预测其中有无山脉、有无水流、有无天空，以实现对其自动打标签的效果。风景图片数据我们这里通过清华云盘链接的形式给出，请各位同学下载后，按照 SubTask 0 中的要求解压到指定位置。


具体而言，在本次作业中我们要实现以下内容：

+ SubTask 0：环境配置与安装（15 p.t.s）
+ SubTask 1：数据预处理（45 p.t.s）
+ SubTask 2：训练框架搭建（60 p.t.s）
+ SubTask 3：结果提交（30 p.t.s）
+ SubTask 4：代码整理与开源（10 p.t.s）



## 环境配置与安装（15 p.t.s）
### 准备 `Python` 环境（10 p.t.s）

我们在前面的课程已经学习过 conda 的使用，你应该可以理解下面指令的作用。

```bash
conda create -n ai python=3.8
conda activate ai
pip install -r requirements.txt
```

然后，执行 `python3 tests/main.py <你的学号>`，这也是之后作业自动评测的脚本执行方法。

### 准备数据集（5 p.t.s）

请从上面清华云盘的链接中下载数据集，然后解压到 `data` 目录下，保证 `data` 目录下直接存在 `train`、`val`、`test` 文件夹与 `LICENSE` 文件。

请阅读 LICENSE 文件，继续进行本作业代表你已知晓并同意 LICENSE 文件中的所有内容。



## 数据预处理（45 p.t.s）

在这一部分我们需要撰写数据预处理的相关函数，你可能会用到 `Pillow`、`NumPy` 等库。

具体来说，我们需要统计 `train` 中 `imgs` 下图片对应的分类标签。`imgs` 中图片的**逐像素**标注位于 `train/labels` 
下，你可以将每张图片认为是一张灰度图，存储了 `0-255` 这 256 个数的其中之一。标签 ID 与标签的对应关系如下：

| 标签 ID         | 标签类别 |
| --------------- | -------- |
| 0, 7            | Mountain |
| 1               | Sky      |
| 2, 3, 8, 16, 20 | Water    |

也就是说，对于一张图片，我们要判断其中有没有山、有没有天空、有没有水，以此来实现对图片打“标签”的效果。接下来我们便要对这些图片及其标签进行预处理，其步骤为：

对于一张图片，如果其中标记为 “Mountain” 像素的个数**超过**了总像素的 20%，我们就认为这张图片中含有 “Mountain”。同理，如果一张图片中标记为 “Sky”、“Water”、“Human Factor” 的像素个数超过了总像素个数的 20%，我们就认为这张图片中含有 “Sky”、“Water”、“Human Factor”。

接下来请阅读并补全 `datasets/dataset_landscape_generator.py` 中的代码，以达到可以产生与 `data/val/file.txt` 相类似的 `data/train/file.txt` 的效果。



## 训练框架搭建



## 结果提交



## 代码整理与开源

请整理你的代码，留足充分的注释后，向大家说明你的代码的使用方式。在向远程仓库推送文件时注意不要提交数据集与你的模型存档点。最后，在[作业仓库](https://github.com/c7w/sast2022-pytorch-training)内新建 Issue，附上自己在 “结果提交” 节的测试结果，然后发送邮件到 cc7w@foxmail.com 说明你的 Issue ID 和学号，方便我们后续统计大家的参与情况与发放服务器作为奖励。



## Appendix

### 标签映射表

```
00 "mountain", "sky", "water", "sea", "rock"
05 "tree", "earth", "hill", "river", "sand"
10 "land", "building", "grass", "plant", "person",
15 "boat", "waterfall", "wall", "pier", "path",
20 "lake", "bridge", "field", "road", "railing",
25 "fence", "ship", "house", "other"
```

