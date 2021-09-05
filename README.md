# 使用说明

## 1. 在本地测试
- 运行```python3 prdict_one.py```即可，默认需要预测的图片路径位于```testImg```文件夹下的```test1.png```
- 运行```python3 predict_folder.py```预测testImg下的所有图片
## 2. 部署到服务器
- 运行```python3 run_a_server.py 8888```即在端口8888部署api，也可以在本地运行测试，如果不填写端口，默认端口为7000
## 3. 通过所部署API进行预测
运行```python3 predict_by_server.py```即可调用所部署的api进行```testImg```文件夹下所有图片的预测

## 4. 打包为可执行文件
```shell
pyinstaller -F run_a_server.py
```
即可打包为可执行文件，打包结束后会在dist文件夹下生成可执行文件。**直接运行不需要python环境**
> 注意！运行打包后的二进制文件时需要将trained_weights.h5放在同级目录下
## 其他说明
- 通过测试，调用本地API预测一张图平均所需约0.05s，服务器1C2G占用内存200~400mb之间
- 我所用的tensorflo和Keras版本分别为：1.15.2和2.3.1
- ```model.py``` 存放网络结构
- trained_weights.h5 训练的权重文件
- 简单curl命令测试
```shell
curl -X POST -F image=@testImg/test1.png 'http://localhost:7000/predict'
```
成功后会返回
```json
{"predictions":0.553,"success":true}
```
本模型所针对的输入图片尺寸大小必须为140x360，没有适配其他图片大小，有能力的可以根据网络结构进行修改从而适应所需。

> 免责声明: 本仓库项目中所涉及的脚本，仅用于测试和学习研究，不保证其合法性，准确性，完整性和有效性，请根据情况自行判断。请勿将本项目的任何内容用于商业或非法途径，否则后果由使用者自负。如果您认为该项目的内容可能涉嫌侵犯其权利，请与我联系，我会尽快删除文件。如果您使用或复制了本仓库项目中的任何内容，则视为您已接受此免责声明。