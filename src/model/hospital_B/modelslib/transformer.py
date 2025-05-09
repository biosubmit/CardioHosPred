import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
 
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
      
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer model for sequence classification
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        # Create mask for Transformer (optional)
        # mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        # Apply Transformer encoder
        transformer_out = self.transformer_encoder(embedded)  # encoder_out: (batch_size, seq_len, hidden_size)
        # Use the output for the last position in the sequence
        out = self.fc(transformer_out[:, -1, :])  # (batch_size, output_size)
        out = self.softmax(out)
        return out
    
    def _generate_square_subsequent_mask(self, sz):
        # Generate mask to prevent attending to future positions
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=1, hidden_size=64, output_size=2, num_layers=2, 
                 num_epochs=10, batch_size=32, learning_rate=0.001, nhead=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nhead = nhead
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model will be initialized in fit
        self.model = None
        self.criterion = None
        self.optimizer = None

    def _build_model(self):
        return TransformerModel(
            self.input_size, 
            self.hidden_size, 
            self.output_size, 
            self.num_layers,
            self.nhead
        ).to(self.device)

    def _prepare_data(self, X, y=None):
        # Reshape input to (batch_size, sequence_length, input_size)
        if len(X.shape) == 2:
            # For Transformer, we need to reshape the data as a sequence
            # If each row is a different feature, we'll treat the features as a sequence
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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )

        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )

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