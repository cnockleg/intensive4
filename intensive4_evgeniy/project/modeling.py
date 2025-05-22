import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from metrics import evaluate_model


class ReviewDataset(Dataset):
    def __init__(self, dataframe, text_column, label_columns, model):
        self.texts = dataframe[text_column].tolist()
        self.labels = dataframe[label_columns].values.astype('float32')
        self.model = model

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        embedding = self.model.encode(text, convert_to_tensor=True)
        label = torch.tensor(self.labels[idx])
        return embedding, label
    

class MultiLabelClassifier(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.hidden = nn.Linear(embedding_dim, embedding_dim)  # Можно оставить ту же размерность
        self.output = nn.Linear(embedding_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return torch.sigmoid(x)


def prepare_dataloaders(train_df, test_df, model_sbert, labels, batch_size=32):
    train_dataset = ReviewDataset(train_df, text_column='comment', label_columns=labels, model=model_sbert)
    test_dataset = ReviewDataset(test_df, text_column='comment', label_columns=labels, model=model_sbert)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(train_loader, embedding_dim, num_labels, epochs=10):
    model = MultiLabelClassifier(embedding_dim, num_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        for embeddings, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model


def predict_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(emb) for emb, _ in data_loader]).cpu().numpy()
        labels = torch.cat([tgt for _, tgt in data_loader]).cpu().numpy()
    return preds, labels


def train_and_evaluate(train_loader, test_loader, embedding_dim, num_labels):
    model = train_model(train_loader, embedding_dim, num_labels)
    train_preds, train_labels = predict_model(model, train_loader)
    test_preds, test_labels = predict_model(model, test_loader)
    return evaluate_model(train_labels, train_preds, test_labels, test_preds)