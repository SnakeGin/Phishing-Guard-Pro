import torch
import torch.nn as nn
import torch.nn.functional as F

class FMPEDModel(nn.Module):
    def __init__(self, input_dim=777, traditional_dim=9):
        """
        初始化 FMPED (Fusion Multi-modal Phishing Email Detection) 模型
        
        Args:
            input_dim (int): 总输入维度 (默认 777)
            traditional_dim (int): 传统特征的维度 (前9维)
        """
        super(FMPEDModel, self).__init__()
        
        self.traditional_dim = traditional_dim
        self.bert_dim = input_dim - traditional_dim # 768
        
        # --- 流 A: 传统特征处理分支 ---
        # 传统特征数值差异大，先做 BatchNorm
        self.trad_bn = nn.BatchNorm1d(self.traditional_dim)
        self.trad_branch = nn.Sequential(
            nn.Linear(self.traditional_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # --- 流 B: BERT语义特征处理分支 ---
        self.bert_branch = nn.Sequential(
            nn.Linear(self.bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # --- 融合层 (Fusion Layer) ---
        # 输入维度 = 流A输出(16) + 流B输出(64) = 80
        combined_dim = 16 + 64
        
        # 融合分类头
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1), # 输出单一数值
            nn.Sigmoid()      # 映射到 0-1 概率
        )
        
    def forward(self, x):
        """
        前向传播
        x: shape (batch_size, 777)
        """
        # 1. 切分数据
        # x_trad: 前9列 (batch_size, 9)
        # x_bert: 后768列 (batch_size, 768)
        x_trad = x[:, :self.traditional_dim]
        x_bert = x[:, self.traditional_dim:]
        
        # 2. 流 A 处理 (含归一化)
        x_trad = self.trad_bn(x_trad)
        out_trad = self.trad_branch(x_trad)
        
        # 3. 流 B 处理
        out_bert = self.bert_branch(x_bert)
        
        # 4. 特征拼接 (Concatenation)
        combined = torch.cat((out_trad, out_bert), dim=1)
        
        # 5. 最终分类
        probability = self.classifier(combined)
        
        return probability

# --- 动态对抗训练逻辑 (Dynamic Adversarial Training) ---

def fgsm_attack(model, data, target, epsilon=0.1):
    """
    生成对抗样本 (Fast Gradient Sign Method)
    用于模拟攻击者试图通过微调邮件特征来绕过检测
    """
    # 创建数据的副本，并允许计算梯度
    data_adv = data.clone().detach().requires_grad_(True)
    
    # 前向传播
    output = model(data_adv)
    loss = F.binary_cross_entropy(output, target)
    
    # 模型梯度清零 (我们只需要对 data 求导)
    model.zero_grad()
    
    # 反向传播，计算 Loss 对 data 的梯度
    loss.backward()
    
    # 获取数据梯度的符号 (sign)
    data_grad = data_adv.grad.data.sign()
    
    # 生成扰动后的数据: x_adv = x + epsilon * sign(grad)
    perturbed_data = data + epsilon * data_grad
    
    # 裁剪，确保特征不会偏离太远 (可选)
    # perturbed_data = torch.clamp(perturbed_data, 0, 1) 
    
    return perturbed_data

def train_step_adversarial(model, optimizer, data, target, epsilon=0.05):
    """
    对抗训练步骤：同时在 原始数据 和 对抗样本 上训练模型
    """
    model.train()
    
    # 1. 正常训练
    optimizer.zero_grad()
    output_clean = model(data)
    loss_clean = F.binary_cross_entropy(output_clean, target)
    loss_clean.backward() # 计算正常梯度
    
    # 2. 生成对抗样本 (基于当前的参数)
    data_adv = fgsm_attack(model, data, target, epsilon)
    
    # 3. 对抗训练 (让模型学会识别这些扰动后的样本)
    output_adv = model(data_adv)
    loss_adv = F.binary_cross_entropy(output_adv, target)
    
    # 这里的 loss_adv 权重可以调整，通常 0.5:0.5 或者 1:1
    # 我们需要在 backward 之前再次 zero_grad 吗？
    # 不，我们希望累积梯度 (Accumulate Gradients) 或者分开 step
    # 简单做法：Loss = Loss_clean + Loss_adv
    
    # 由于上面已经 backward 了一次，这里我们需要一种合并方式。
    # 更标准的写法是把两次 Loss 加起来一起 backward
    optimizer.zero_grad() # 重置
    
    # 重新计算流程以合并 Loss
    output_clean = model(data)
    loss_clean = F.binary_cross_entropy(output_clean, target)
    
    output_adv = model(data_adv) # data_adv 已经不需要梯度了
    loss_adv = F.binary_cross_entropy(output_adv, target)
    
    total_loss = 0.5 * loss_clean + 0.5 * loss_adv
    total_loss.backward()
    
    optimizer.step()
    
    return total_loss.item()

# # --- 测试与验证代码 ---
# if __name__ == "__main__":
#     # 1. 初始化模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = FMPEDModel().to(device)
#     print("模型架构:\n", model)
    
#     # 2. 模拟输入数据 (Batch size = 4, Dim = 777)
#     # 假设前4个样本：2个正常(0)，2个钓鱼(1)
#     dummy_input = torch.randn(4, 777).to(device)
#     dummy_labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]]).to(device)
    
#     # 3. 测试前向传播
#     with torch.no_grad():
#         model.eval()
#         probs = model(dummy_input)
#         print("\n初始预测概率 (未训练):")
#         print(probs.cpu().numpy())
        
#     # 4. 测试对抗训练步骤
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     print("\n开始对抗训练测试...")
#     for i in range(5):
#         loss = train_step_adversarial(model, optimizer, dummy_input, dummy_labels)
#         print(f"Epoch {i+1}, Loss: {loss:.4f}")