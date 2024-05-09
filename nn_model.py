import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from data_analyze import cabin_to_letter, fillna_by_median, feature_encoding, features2tensor

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

def main():

    
    fillna_by_median('./Titanic_data/train.csv', './Titanic_data/train_filled.xlsx', 'Age')
    fillna_by_median('./Titanic_data/test.csv', './Titanic_data/test_filled.xlsx', 'Age')

    cabin_to_letter('./Titanic_data/train_filled.xlsx')
    cabin_to_letter('./Titanic_data/test_filled.xlsx')

    feature_encoding('./Titanic_data/train_filled.xlsx', ['Sex', 'Embarked', 'Cabin_Letter'])
    feature_encoding('./Titanic_data/test_filled.xlsx', ['Sex', 'Embarked', 'Cabin_Letter'])

    df = pd.read_excel('./Titanic_data/train_filled.xlsx')

    numpy_labels = df['Survived'].to_numpy()  # 使用to_numpy()明确转换为numpy数组
    # 将numpy数组转换为PyTorch tensor
    train_labels = torch.tensor(numpy_labels, dtype=torch.float32)

    lt_fields =  ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Letter']

    train_features = features2tensor('./Titanic_data/train_filled.xlsx',lt_fields)
    test_features = features2tensor('./Titanic_data/test_filled.xlsx',lt_fields)

    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(test_features)

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BinaryClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model

    for epoch in range(100):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './Titanic_data/model_weights.pth')
    model.load_state_dict(torch.load('./Titanic_data/model_weights.pth'))

    # Evaluation mode
    model.eval()
    df = pd.read_excel('./Titanic_data/test_filled.xlsx')
    passenger_ids = df['PassengerId']
    all_predictions = []
    all_passenger_ids = []
    index = 0

    with torch.no_grad():
        for inputs in test_loader:

            outputs = model(inputs[0])
            predictions = (outputs.squeeze() > 0.5).int()

            all_predictions.extend(predictions.tolist())

            batch_size = inputs[0].size(0)
            batch_passenger_ids = passenger_ids[index:index + batch_size]
            all_passenger_ids.extend(batch_passenger_ids)

            index += batch_size

        assert index == len(passenger_ids), "Not all PassengerIds were processed"

    df = pd.DataFrame({'PassengerId': all_passenger_ids, 'Survived': all_predictions})
    df.to_csv('./Titanic_data/test_prediction.csv', index=False)

    return 0

if __name__ == '__main__':
    main()
