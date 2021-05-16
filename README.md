# fcanet-Paddle
基于Paddle框架的fcanet复现

## fcanet

本项目基于paddlepaddle框架复现fcanet，并参加百度第三届论文复现赛，将在2021年5月15日比赛完后提供AIStudio链接～敬请期待

参考项目：

[frazerlin-fcanet](https://github.com/frazerlin/fcanet)

## 数据准备
**本项目已挂载论文所使用的数据集，对于`tgz`及`tar`文件需要利用以下命令解压**
```
tar -xvf benchmark.tgz
tar xvf VOCtrainval_11-May-2012.tar
```
整个工程具有以下目录结构
```
/home/aistudio
|───Data(数据集)
└───────benchmark_RELEASE
└───────VOCdevkit
└───────GrabCut
└───────Berkeley
└───fcanet(代码文件)
└───InitialPaddleModel(初始化权重)
```
## 训练
The official PyTorch implementation of CVPR 2020 paper ["Interactive Image Segmentation with First Click Attention"](http://mftp.mmcheng.net/Papers/20CvprFirstClick.pdf).
并未提供训练代码。通过邮件联系作者，作者由于企业合作项目原因，合作结束后会将会提供训练代码



## 测试

[模型下载](https://pan.baidu.com/s/1XDRSVPdkyqW1WdMScPDiIQ)

提取码：2ira

[AIStudio链接](https://aistudio.baidu.com/aistudio/projectdetail/1892923?channel=0&channelType=0&shared=1)

### 验证集测试
```
python fcanet/evaluate.py --backbone [resnet/res2net] --dataset [GrabCut,Berkeley,DAVIS(not exists in this repo),VOCdevkit] (--sis)
```

如下图所示，**默认的`backbone`均为101**

#### resnet101测试示例

<img src="https://ai-studio-static-online.cdn.bcebos.com/92a234a62c7848caa9fa57bc246538bdad2def1d12dc425680d989c237e5e9ad" width="700"/>

#### res2net101测试示例

<img src="https://ai-studio-static-online.cdn.bcebos.com/92a234a62c7848caa9fa57bc246538bdad2def1d12dc425680d989c237e5e9ad" width="700"/>

|backbone|dataset|mNoC|mIoU-NoC|
|--------|--------|--------|--------|
|resnet101|Berkeley|4.23|[0.    0.728 0.854 0.885 0.912 0.915 0.926 0.935 0.939 0.935 0.94  0.943  0.942 0.944 0.945 0.945 0.947 0.947 0.948 0.947 0.949]|
|resnet101|GrabCut|2.24|[0.    0.78  0.87  0.923 0.944 0.95  0.956 0.966 0.964 0.971 0.971 0.971  0.975 0.977 0.978 0.979 0.978 0.978 0.979 0.979 0.979]|
|resnet101|VOC2012|2.9810329734461627|[0.    0.715 0.838 0.885 0.909 0.926 0.937 0.945 0.951 0.957 0.962 0.964 0.967 0.969 0.971 0.973 0.974 0.976 0.977 0.978 0.979]|
|res2net101|Berkeley|3.98|[0.    0.788 0.872 0.901 0.921 0.93  0.933 0.938 0.938 0.943 0.943 0.943 0.943 0.945 0.947 0.948 0.949 0.949 0.95  0.951 0.95 ]|
|res2net101|GrabCut|2.16|[0.    0.819 0.877 0.927 0.916 0.931 0.948 0.96  0.966 0.967 0.969 0.971 0.973 0.976 0.977 0.976 0.978 0.977 0.98  0.977 0.979]|
|res2net101|VOC2012|2.793988911584476|[0.    0.757 0.841 0.882 0.908 0.925 0.937 0.945 0.952 0.958 0.963 0.966 0.968 0.971 0.973 0.974 0.976 0.977 0.978 0.98  0.98 ]|

### 可视化测试
**利用`annotator.py`可以实现可视化操作，感兴趣的读者可是利用Qt实现UI程序，实现效果如下所示**

***需要注意的是，AIStudio环境暂不支持这种可视化方式，你需要将此仓库部署到本地运行，你可能需要修改代码文件中的路径***
```python
python fcanet/annotator.py --backbone res2net --input fcanet/test.jpg --output test_mask.jpg
```
<img src="https://img-blog.csdnimg.cn/20210503140453264.gif" width="700"/>

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        | 郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| 主页        | [Deep Hao的主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
