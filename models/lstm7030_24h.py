import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

file_path = 'data/processed/data_for_models.pkl'
data = pd.read_pickle(file_path)

target = data['count'].values.astype(float)  #using number of departures as target                                                                                                              

#normalize the data                                                                                                                                                                             
scaler = MinMaxScaler(feature_range=(-1, 1)) #minmax scaler does not assume the data has a specific distribution shape                                                                          
target_normalized = scaler.fit_transform(target.reshape(-1, 1))

#create sequences                                                                                                                                                                               
def create_sequences(target, seq_length):
    xs = []
    ys = []
    for i in range(len(target) - seq_length):
        x = target[i:i + seq_length]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 24  #trying sequence length of 24 hours                                                                                                                                            
X, y = create_sequences(target_normalized, seq_length)

#perform train/test split (70/30)                                                                                                                                                               
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#convert to pytorch tensors                                                                                                                                                                     
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

#make a custom dataset for a DataLoader                                                                                                                                                         
class BikeshareDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BikeshareDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        #LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        #take the last timestep's output and pass to fully connected layer
        out = self.fc(out[:, -1, :])
        return out

#instantiate model, loss, optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

#evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    test_loss = criterion(y_pred_test, y_test)
    print(f'Test Loss: {test_loss.item():.6f}')

#inverse transform to get predicted values from test
y_pred_test_703024 = scaler.inverse_transform(y_pred_test.numpy())
y_test_703024 = scaler.inverse_transform(y_test.numpy())
