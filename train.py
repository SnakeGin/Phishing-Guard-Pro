import os
# 设置 Hugging Face 国内镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 导入你的核心模块
from feature_extractor import PhishingFeatureExtractor
from model_architecture import FMPEDModel, train_step_adversarial

# --- 配置参数 ---
CONFIG = {
    "dataset_repo": "ealvaradob/phishing-dataset",
    "dataset_subset": "combined_reduced", # ⚠️ 关键：使用精简版以保证数据平衡
    "max_samples": None,       # 设为 None 则使用全量数据，设为数字(如 5000)则用于快速测试
    "batch_size": 32,
    "epochs": 5,               # 真实数据量大，Epoch 可以适当减少
    "learning_rate": 2e-5,     # BERT 微调通常需要很小的学习率
    "model_save_path": "fmped_model.pth",
    "feature_cache_path": "features_cache_real.npz"
}

def load_dataset_strictly():
    """
    严格参照数据集文档的加载方式
    文档要求: dataset = load_dataset(..., "combined_reduced", ...)
    """
    print(f"🌍 正在连接 HuggingFace 下载数据集: {CONFIG['dataset_repo']} [{CONFIG['dataset_subset']}] ...")
    print("ℹ️  提示: 文档建议使用 'combined_reduced' 以避免 URL 数据带来的类别偏差。")
    
    try:
        # 1. 加载数据集 (根据文档，这是一个 DatasetDict，只有 'train' 分支)
        dataset = load_dataset(
            CONFIG['dataset_repo'], 
            CONFIG['dataset_subset'], 
            trust_remote_code=True
        )
        
        # 2. 转换为 Pandas DataFrame
        df = dataset['train'].to_pandas()
        
        print("📊 数据集加载成功，正在检查结构...")
        # 文档说明结构: columns=['text', 'label'], label: 1(Phishing), 0(Benign)
        
        # 3. 简单的数据清洗 (文档说已经去重去空，这里做个兜底检查)
        initial_len = len(df)
        df = df.dropna(subset=['text', 'label'])
        
        # 4. 确保标签是数字格式
        df['label'] = df['label'].astype(float)
        
        print(f"✅ 数据准备就绪: {len(df)} 条样本 (原始: {initial_len})")
        print(f"   - 钓鱼样本 (1): {len(df[df['label']==1])}")
        print(f"   - 正常样本 (0): {len(df[df['label']==0])}")
        
        # 5. 采样限制 (如果配置了 max_samples)
        if CONFIG['max_samples'] and len(df) > CONFIG['max_samples']:
            print(f"✂️ 仅使用前 {CONFIG['max_samples']} 条数据进行训练 (配置限制)...")
            # 保持分层采样以维持平衡
            df = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), CONFIG['max_samples'] // 2), random_state=42)
            )
            # 打乱
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df['text'].tolist(), df['label'].tolist()

    except Exception as e:
        print(f"❌ 数据加载严重错误: {e}")
        print("💡 请检查网络连接或 huggingface 库版本 (pip install --upgrade datasets)")
        exit(1)

def extract_features_robust(texts, labels, extractor):
    """
    特征提取逻辑调整：
    该数据集是混合类型的 (URL, SMS, HTML, Email)，
    我们需要让提取器尽可能多地挖掘信息。
    """
    if os.path.exists(CONFIG['feature_cache_path']):
        print(f"💾 发现缓存特征 '{CONFIG['feature_cache_path']}'，正在加载...")
        data = np.load(CONFIG['feature_cache_path'])
        if len(data['y']) == len(labels):
            print("✅ 缓存校验通过，跳过 BERT 提取。")
            return data['X'], data['y']
        else:
            print("⚠️ 缓存数量不匹配，重新提取...")

    print("🚀 开始多模态特征提取 (Pipeline)...")
    features_list = []
    valid_labels = []
    
    # 进度条
    for i, content in enumerate(tqdm(texts, desc="Processing")):
        try:
            content_str = str(content)
            
            # --- 关键策略调整 ---
            # 因为数据集中有些行是纯 HTML 代码，有些是纯 URL，有些是纯文本。
            # 我们将 content 同时传给 raw_text 和 html_content。
            # 1. 如果它是 HTML，BS4 会解析出 tag 特征。
            # 2. 如果它是 URL，正则会提取出 URL 特征。
            # 3. BERT 会读取原文提取语义。
            
            # 截断过长文本防止内存爆炸 (特别是 HTML 代码可能很长)
            truncated_content = content_str[:10000] 
            
            result = extractor.process_email(
                raw_text=truncated_content, 
                html_content=truncated_content 
            )
            
            features_list.append(result['fused_vector'])
            valid_labels.append(labels[i])
            
        except Exception as e:
            # 容错处理，打印错误但不停机
            # print(f"⚠️ Error processing sample {i}: {e}")
            continue

    X = np.array(features_list, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.float32)
    
    print(f"💾 缓存特征到 {CONFIG['feature_cache_path']} ...")
    np.savez(CONFIG['feature_cache_path'], X=X, y=y)
    
    return X, y

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    # average='binary' 适用于二分类
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    return acc, p, r, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ 计算设备: {device}")
    
    # 1. 加载数据 (Combined Reduced Dataset)
    texts, labels = load_dataset_strictly()
    
    # 2. 初始化提取器
    print("🧠 初始化 BERT 特征提取器...")
    extractor = PhishingFeatureExtractor()
    
    # 3. 提取特征 (含缓存机制)
    X_data, y_data = extract_features_robust(texts, labels, extractor)
    
    # 4. 划分数据集 (文档建议: 80% Train, 20% Test)
    # 使用 stratify 确保训练集和测试集的黑白样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    
    # 5. 封装 DataLoader
    train_ds = TensorDataset(torch.tensor(X_train).to(device), torch.tensor(y_train).unsqueeze(1).to(device))
    test_ds = TensorDataset(torch.tensor(X_test).to(device), torch.tensor(y_test).unsqueeze(1).to(device))
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 6. 模型初始化
    print("🏗️ 构建 FMPED 模型...")
    model = FMPEDModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.BCELoss() # 二分类交叉熵
    
    # 7. 训练循环
    print(f"⚔️ 开始全量训练 (Epochs: {CONFIG['epochs']}) ...")
    best_f1 = 0.0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            # 混合使用普通训练和对抗训练
            # 这里我们每步都使用对抗训练来增强模型对微小扰动的鲁棒性
            loss = train_step_adversarial(model, optimizer, batch_x, batch_y, epsilon=0.03)
            total_loss += loss
            
        avg_loss = total_loss / len(train_loader)
        
        # 验证
        acc, prec, rec, f1 = evaluate(model, test_loader, device)
        
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"    🌟 新的最佳模型已保存 (F1: {best_f1:.4f})")

    print("\n✅ 训练结束。请重启 main.py 以加载新模型。")

if __name__ == "__main__":
    main()