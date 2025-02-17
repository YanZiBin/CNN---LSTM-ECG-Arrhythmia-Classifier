import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子和设备
torch.manual_seed(222226205227)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 自定义数据集类用于批量加载心电图数据
class ECGDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)
        self.x = torch.FloatTensor(data.iloc[:, :-1].values)
        self.y = torch.LongTensor(data.iloc[:, -1].values)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 混合CNN和LSTM的模型
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        
        # CNN特征提取层
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM序列处理层
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        
        # 全连接分类层
        self.fc = nn.Sequential(
            nn.Linear(256 * 23, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 5)
        )
        
    def forward(self, x):
        # 调整输入维度
        x = x.unsqueeze(1)
        
        # CNN特征提取
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 准备LSTM输入
        x = x.permute(0, 2, 1)
        
        # LSTM处理
        x, _ = self.lstm(x)
        
        # 展平特征
        x = x.reshape(x.size(0), -1)
        
        # 全连接分类
        x = self.fc(x)
        return x

# 模型训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

# 模型测试函数
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(test_loader), 100. * correct / total, all_preds, all_labels

# 主程序
def main():
    # 加载训练和测试数据
    train_dataset = ECGDataset('E:/Yan.doument2/programming exercises/python/Test/data/mitbih_train.csv/mitbih_train.csv')
    test_dataset = ECGDataset('E:/Yan.doument2/programming exercises/python/Test/data/mitbih_test.csv/mitbih_test.csv')
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 初始化模型和优化器
    model = CNNLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # 训练参数设置
    epochs = 30
    best_acc = 0
    
    # 存储训练过程指标
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # 开始训练循环
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, preds, labels = test_model(model, test_loader, criterion, device)
        
        # 记录训练指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step(test_loss)
        
        print(f'轮次: {epoch+1}/{epochs}')
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
        print('-' * 60)
        
        # 保存最佳模型结果
        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = preds
            best_labels = labels
    
    # 绘制训练过程图表
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='训练损失')
    plt.plot(epochs_range, test_losses, 'r-', label='测试损失')
    plt.title('损失随轮次的变化')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, 'b-', label='训练准确率')
    plt.plot(epochs_range, test_accs, 'r-', label='测试准确率')
    plt.title('准确率随轮次的变化')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 输出最终评估结果
    print(f'\n最佳测试准确率: {best_acc:.2f}%')
    
    # 绘制混淆矩阵
    cm = confusion_matrix(best_labels, best_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    # 输出详细分类报告
    print('\n分类报告:')
    print(classification_report(best_labels, best_preds))

if __name__ == '__main__':
    main()
