# Titanic神经网络二分类

## 目录结构

├── Titanic_data                              
│   ├── train.csv                             原始训练集
│   ├── test.csv                              原始测试集
│   ├── model_weights.pth                     神经网络模型权重
│   ├── test.xlsx                             经过预处理后的测试集
│   ├── train_filled.xlsx                     经过预处理后的训练集
│   └── test_prediction.csv                   关于数据库操作的文件夹
├── data_analyze.py                           特征分析，数据预处理，包括空值填充，将非数值类型的特征进行数字编码等
├── data_preprocess.py                        创建数据集、dataloader
├── nn_model.py                               神经网络模型的定义、训练、预测
## 使用方法

## 启动服务
python nn_model.py
