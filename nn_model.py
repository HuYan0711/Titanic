import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_analyze import cabin_to_letter, fillna_by_median, feature_encoding, features2tensor


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(8, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()


def main():

    sv_path = './Titanic_data/test_prediction.csv'
    use_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Letter']
    batch_size = 32


    train_df = fillna_by_median('./Titanic_data/train.csv', 'Age')
    test_df = fillna_by_median('./Titanic_data/test.csv', 'Age')

    train_df = cabin_to_letter(train_df)
    test_df = cabin_to_letter(test_df)

    encoding_features = ['Sex', 'Embarked', 'Cabin_Letter']
    train_df = feature_encoding(train_df, encoding_features)
    test_df = feature_encoding(test_df, encoding_features)

    train_labels = torch.tensor(train_df['Survived'].to_numpy(), dtype=torch.float32)

    train_features = features2tensor(train_df, use_features)
    test_features = features2tensor(test_df, use_features)

    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(test_features)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = BinaryClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    for epoch in range(100):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), './Titanic_data/model_weights.pth')

    model.load_state_dict(torch.load('./Titanic_data/model_weights.pth'))

    # Evaluation mode
    model.eval()
    passenger_ids = test_df['PassengerId']
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
    df.to_csv(sv_path, index=False)


if __name__ == '__main__':
    main()
