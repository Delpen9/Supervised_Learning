# Standard Libraries
import numpy as np
import pandas as pd

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Model Evaluation
from sklearn.metrics import roc_auc_score, accuracy_score


class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def tune_neural_network(
    train_loader, val_loader, input_size, num_epochs=10, learning_rate=0.001, multiclass=False, num_classes=2
):
    print(multiclass)
    print(num_classes)

    pos = torch.tensor([label for _, label in train_loader.dataset]).sum().item()
    neg = len(train_loader.dataset) - pos
    class_weights = torch.tensor(
        [1 / neg, 1 / pos]
    )
    class_weights = (
        class_weights / class_weights.sum()
    )

    if multiclass == False:
        model = BinaryClassifier(input_size)
    else:
        model = MultiClassClassifier(input_size, num_classes)

    # Use the weighted loss function
    if multiclass == False:
        criterion = nn.BCELoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training loop
        for inputs, labels in train_loader:
            outputs = model(inputs)
            if multiclass == False:
                loss = criterion(outputs, labels.unsqueeze(1).to(torch.long))
            else:
                loss = criterion(outputs, labels.to(torch.long))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                if multiclass == False:
                    loss = criterion(outputs, labels.unsqueeze(1).to(torch.long))
                else:
                    loss = criterion(outputs, labels.to(torch.long))
                val_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
        )

    return model


def evaluate_model(model, test_loader, num_classes=2):
    model.eval()
    true_labels = []
    predicted_probs = []
    predicted_labels = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)

            if num_classes == 2:
                predicted = torch.round(outputs)
                predicted_labels.extend(predicted.numpy())
                predicted_probs.extend(outputs.numpy())
                true_labels.extend(labels.numpy())
            else:
                probs = softmax(outputs)
                predicted = torch.argmax(outputs, dim=1)
                predicted_labels.extend(predicted.numpy())
                predicted_probs.extend(probs.numpy())
                true_labels.extend(labels.numpy())

    auc = roc_auc_score(true_labels, predicted_probs, multi_class='ovr', average='macro')
    accuracy = accuracy_score(true_labels, predicted_labels)

    return auc, accuracy


if __name__ == "__main__":
    data = pd.read_csv("../../data/auction_verification_dataset/data.csv")

    X = data.iloc[:, :-2].copy()
    y = data.iloc[:, -2].copy().astype(int)

    dataset = TensorDataset(
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32),
    )

    train_size = int(0.8 * len(dataset))
    temp_size = len(dataset) - train_size
    train_dataset, temp_dataset = random_split(dataset, [train_size, temp_size])

    val_size = int(0.5 * temp_size)
    test_size = temp_size - val_size
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

    input_size = X.shape[1]
    num_epochs = 200

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    best_model = tune_neural_network(train_loader, val_loader, input_size, num_epochs)

    auc, accuracy = evaluate_model(best_model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
