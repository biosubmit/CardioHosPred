import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# GRU model for sequence classification
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), 
                        self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Take the last time step
        out = self.softmax(out)
        return out

class GRUClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=1, hidden_size=64, output_size=2, num_layers=1, 
                 num_epochs=10, batch_size=32, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model will be initialized in fit
        self.model = None
        self.criterion = None
        self.optimizer = None

    def _build_model(self):
        return GRUModel(self.input_size, self.hidden_size, 
                       self.output_size, self.num_layers).to(self.device)

    def _prepare_data(self, X, y=None):
        # Reshape input to (batch_size, sequence_length, input_size)
        if len(X.shape) == 2:
            X = X.values.reshape(X.shape[0], -1, self.input_size)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if y is not None:
            y_tensor = torch.LongTensor(y.values.ravel()).to(self.device)
            return X_tensor, y_tensor
        return X_tensor

    def fit(self, X, y):
        # Initialize model, criterion, and optimizer
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.learning_rate)

        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size,
                                                 shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
        
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._prepare_data(X)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._prepare_data(X)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy() 