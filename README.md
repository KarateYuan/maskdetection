# 人脸佩戴口罩识别
## 项目环境
- Windows 10
- Python >= 3.8 

运行项目所需的库可通过以下命令安装
```
pip install -r requirements.txt
```

## 代码详解
### 程序主入口
```
tensorflow_infer.py
```
在终端通过运行`python tensorflow_infer.py`则会捕获本计算机的摄像头，从而进行人脸是否佩戴口罩检测。
推理的基本流程大概是:
- 从摄像头读取图片进行图片预处理(如通道转换BGR->RGB,resize成模型输入的像素，归一化)
- 进入模型得到输出结果
- 后处理(如根据anchor对输出结果进行解码，得到的结果可能是多个框和分值，进行single_NMS只得到一个检测框，将检测框绘制到图像上)

### 重点看的文件
- tensorflow_infer.py
- utils/anchor_generator.py             (生成anchor)
- utils/anchor_decode.py                (根据anchor解码出检测框)
- utils/nms.py                          (非极大值抑制的实现)

### 训练
本仓库中的代码没有涉及模型是如何训练的，但是原作者是通过pytorch训练的，因此学习时需要只需要了解模型的网络结构即可。
