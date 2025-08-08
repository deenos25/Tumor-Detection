import cv2
import numpy as np
import pandas as pd
from preprocess import TumorImageProcessing
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# Get data frame processed from extract_data.py
file_path = "C:\\Users\\deeno\\.cache\\processed_image_data.pkl.gz"
df = pd.read_pickle(file_path)

# The data frame has Train & Val data
df_train = df[df['set_type'] == 'Train']
df_val = df[df['set_type'] == 'Val']

# Original dataset had no test sample from training (80%, 10%, 10%)
df_test = df_train.sample(n=len(df_val), random_state=0)
index_to_drop = df_test.index
df_train = df_train.drop(index=df_test.index)

# Prepare input/output datasets
def prepare_dataset(dataframe, n_tumors_max):
    inputs, outputs = [], []
    for i in range(len(dataframe)):
        sample = dataframe.iloc[i]
        data_prep = TumorImageProcessing(sample['image'], sample['box_coor'], sample['class'], n_tumors_max)
        inputs.append(data_prep.resize_and_normalize_image())
        outputs.append(data_prep.create_output_array())
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    outputs = torch.tensor(np.array(outputs).astype(np.float32))
    return inputs, outputs

max_tumor_count = df['box_coor'].apply(len).max()
X_train, y_train = prepare_dataset(df_train, max_tumor_count)
X_val, y_val = prepare_dataset(df_val, max_tumor_count)
X_test, y_test = prepare_dataset(df_test, max_tumor_count)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
batch = 16
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

from simple_model import SimpleCNN
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

from loss_function import custom_loss

optimizer = optim.Adam(model.parameters(), lr=1e-3)
from sklearn.metrics import accuracy_score

num_epochs = 16
thr = np.inf
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = custom_loss(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    total = 0
    correct_tumor, correct_class = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = custom_loss(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            total += y_batch.size(0)
            outputs = outputs.cpu().detach().numpy()
            y_batch = y_batch.cpu().detach().numpy()
            mask = (y_batch[:, :, 0] == 1) | (outputs[:, :, 0] >= 0.5)
            print('Binary', accuracy_score(np.round(outputs[:, :, 0][mask]), y_batch[:, :, 0][mask]))
            print('NRMSE', np.sqrt(np.mean(np.square(outputs[:, :, 1:5][mask] - y_batch[:, :, 1:5][mask]))) / np.mean(y_batch[:, :, 1:5][mask]))
            print('Multi', accuracy_score(outputs[:, :, 5:][mask].argmax(axis=1), y_batch[:, :, 5][mask]))
    val_loss /= len(val_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
    if val_loss < thr:
        torch.save(model.state_dict(), 'best_model.pth')
        thr = val_loss

model.eval()
with torch.no_grad():
    sample = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(0).to(device)
    pred = model(sample)[0]
pred = pred.cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()
mask = (y_test[0][:, 0] == 1) | (pred[:, 0] >= 0.5)
ref_boxes, pred_boxes = y_test[0][mask][:, 1:5], pred[mask][:, 1:5]
color_image = cv2.cvtColor(X_test[0].numpy(), cv2.COLOR_GRAY2RGB)
for rb in ref_boxes:
    x, y, w, h = rb * 512  # ensure integers
    x, y, w, h = int(x), int(y), int(w/2), int(h/2)
    cv2.rectangle(color_image, (x-w, y-h), (x + w, y + h), (0, 255, 0), 2)
for rb in pred_boxes:
    x, y, w, h = rb * 512  # ensure integers
    x, y, w, h = int(x), int(y), int(w/2), int(h/2)
    cv2.rectangle(color_image, (x-w, y-h), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('test', color_image)

import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt