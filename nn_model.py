import logging

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_analyze import features2tensor, process_features

logging.basicConfig(filename='./Titanic_data/training.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M')


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


def train(model,
          train_loader,
          val_loader,
          weight_path,
          max_epoch,
          lr,
          threshold=0.5):
    # Train model
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    best_acc = 0.0
    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_epoch_loss = running_loss / len(train_loader)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_outputs = model(val_inputs)
                predicted = (val_outputs >= threshold).float()
                correct += (predicted == val_labels.float()).sum().item()
                total += val_labels.size(0)
        accuracy = 100 * correct / total
        logging.info(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), weight_path)


def predict(model, weight_path, test_df, test_loader):
    model.load_state_dict(torch.load(weight_path))
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
    return df


def main():
    train_path = './Titanic_data/train.csv'
    test_path = './Titanic_data/test.csv'

    weight_path = './Titanic_data/model_weights.pth'
    results_save_path = './Titanic_data/test_prediction.csv'

    encoding_features = ['Sex', 'Embarked', 'Cabin_Letter']
    use_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Letter']

    split_ratio = 0.2
    lr = 0.001
    batch_size = 32
    max_epoch = 100

    train_df = process_features(train_path, encoding_features, column_name='Age')
    test_df = process_features(test_path, encoding_features, column_name='Age')

    train_labels = torch.tensor(train_df['Survived'].to_numpy(), dtype=torch.float32)

    train_features = features2tensor(train_df, use_features)
    test_features = features2tensor(test_df, use_features)

    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=split_ratio,
                                                      random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(test_features)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = BinaryClassifier()

    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          weight_path=weight_path,
          max_epoch=max_epoch,
          lr=lr,
          threshold=0.5)

    pred_df = predict(model=model,
                      weight_path=weight_path,
                      test_df=test_df,
                      test_loader=test_loader)
    pred_df.to_csv(results_save_path, index=False)


if __name__ == '__main__':
    main()
