import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from data_preprocess import train_loader, val_loader, test_loader

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(8, 20)  # 8 features，20 nodes hidden layers
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(20, 1)  # 1 output node layer for binary classify
        self.sigmoid = nn.Sigmoid()  # 对于二分类问题，输出层通常使用Sigmoid激活函数

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()


# initialize model and optimizer
model = BinaryClassifier()
criterion = nn.BCELoss()  # 二分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# Train model

for epoch in range(100):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())  # labels需要是float类型以匹配BCELoss的期望输入

        # 反向传播和优化
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 使用优化器更新权重
torch.save(model.state_dict(), './Titanic_data/model_weights.pth')

model.load_state_dict(torch.load('./Titanic_data/model_weights.pth'))

# Evaluation mode
model.eval()

passenger_ids = data_test_xlsx['PassengerId']
all_predictions = []  # For save prediction results
all_passenger_ids = []  # For save PassengerId
index = 0

with torch.no_grad():
    for inputs in test_loader:
        # The data in a dataloader is usually a tuple (feature, label), but if no label is defined, the input type is list [tensor]
        outputs = model(inputs[0])
        predictions = (
                    outputs.squeeze() > 0.5).int()  # Binary prediction, assuming that the output is greater than 0.5, it is considered positive; otherwise, it is considered negative

        all_predictions.extend(predictions.tolist())

        # take current batch's IDs from passenger_ids list
        batch_size = inputs[0].size(0)
        batch_passenger_ids = passenger_ids[index:index + batch_size]  # 提取当前批次的PassengerId
        all_passenger_ids.extend(batch_passenger_ids)

        index += batch_size

    assert index == len(passenger_ids), "Not all PassengerIds were processed"

df = pd.DataFrame({'PassengerId': all_passenger_ids, 'Survived': all_predictions})
df.to_csv('./Titanic_data/test_prediction.csv', index=False)
