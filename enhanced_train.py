import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 1. Enhanced Dataset Class with Augmentation ----------------------------
class SquatDataset(Dataset):
    def __init__(self, data_root, max_seq_len=100):
        self.max_seq_len = max_seq_len
        self.data = []
        self.labels = []
        
        # Load and flatten sequences properly
        for label, class_name in enumerate(["Valid", "Invalid"]):
            class_path = os.path.join(data_root, class_name)
            for video_dir in sorted(os.listdir(class_path), key=int):
                video_path = os.path.join(class_path, video_dir)
                sequence = []
                for frame_file in sorted(os.listdir(video_path), 
                                      key=lambda x: int(x.split('.')[0])):
                    frame_data = np.load(os.path.join(video_path, frame_file))
                    
                    # Ensure proper flattening: (33,4) -> (132,)
                    if frame_data.ndim == 1:
                        frame_data = frame_data.reshape(33, 4)
                    sequence.append(frame_data.flatten())  # Crucial fix
                
                self.data.append(np.array(sequence))
                self.labels.append(label)

        self.max_seq_len = max(len(seq) for seq in self.data)
        print(f"Auto-detected max_seq_len: {self.max_seq_len}") #for debug

    def __len__(self): 
        return len(self.labels)
    
    def __getitem__(self, idx):
        keypoints = self.data[idx].copy()
        orig_len = keypoints.shape[0]
        
        # Pad/Truncate to fixed length
        if orig_len < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - orig_len, 132))  # Maintain 2D shape
            keypoints = np.vstack((keypoints, pad))
        else:
            keypoints = keypoints[:self.max_seq_len]
        
        # Augmentation with proper reshaping
        if np.random.rand() > 0.5:
            # Reshape to (seq_len, 33, 4) for rotation
            keypoints_3d = keypoints.reshape(-1, 33, 4)
            
            # Apply rotation
            angle = np.random.uniform(-5, 5)
            rot_mat = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])
            keypoints_3d[..., :2] = np.dot(keypoints_3d[..., :2], rot_mat)
            
            # Flatten back to 2D: (seq_len, 132)
            keypoints = keypoints_3d.reshape(-1, 132)
        
        return torch.tensor(keypoints, dtype=torch.float32), self.labels[idx]

# 2. Advanced Model Architecture -----------------------------------------
class SquatClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=132,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),  # 2*hidden_size due to bidirectional
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2))

    def forward(self, x):
        # LSTM Layer
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 256)
        
        # Attention Calculation
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Classification
        return self.fc(context)

# 3. Enhanced Training Loop ----------------------------------------------
def train():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SquatDataset("C:\\Users\IzzatHamizan\Documents\WorkoutFormChecker\Exercise-Form-Checker-main\Assets\squatData\Squat_Data", max_seq_len=100)
    train_size = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_size, len(dataset)-train_size])
    
    # Data Loaders
    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: ( #use collate funcion to ensure consistency
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch])
        ),
        num_workers=0  # Disable multiprocessing for debugging
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=64, 
        num_workers=0
    )
    
    # Model Setup
    model = SquatClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    
    # Training Parameters
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'acc': [], 'f1': []}
    
    for epoch in range(100):
        # Training Phase
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # Shape: (batch, seq_len, features)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Validation Phase
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels.to(device)).item()
                val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate Metrics
        train_acc = accuracy_score(all_labels, all_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        # Update Learning Rate
        scheduler.step(val_loss/len(val_loader))
        
        # Save Best Model
        current_val_loss = val_loss/len(val_loader)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Print Progress
        print(f"\nEpoch {epoch+1:03d}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}")
        print(f"  Val Loss: {current_val_loss:.4f} | Acc: {val_acc:.2f} | F1: {val_f1:.2f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early Stopping
        if early_stop_counter >= 5:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    train()