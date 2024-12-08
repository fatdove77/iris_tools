import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from glob import glob
import json
from datetime import datetime

class InceptionScoreCalculator:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        print("Initializing Inception-v3 model...")
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = nn.Identity()  # 移除最后的全连接层
        self.inception_model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.inception_model = self.inception_model.to(self.device)
        
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def load_and_preprocess_images(self, image_paths):
        n_batches = int(math.ceil(len(image_paths) / self.batch_size))
        for i in range(n_batches):
            batch_paths = image_paths[i * self.batch_size:(i + 1) * self.batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.preprocess(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    continue
                
            if batch_images:
                yield torch.stack(batch_images).to(self.device)

    def calculate_activation_statistics(self, image_paths):
        preds_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.load_and_preprocess_images(image_paths),
                            total=int(math.ceil(len(image_paths) / self.batch_size)),
                            desc="Processing images"):
                try:
                    # 确保batch的维度正确
                    if batch.ndim != 4:
                        print(f"Unexpected batch dimension: {batch.shape}")
                        continue
                        
                    # 获取特征
                    features = self.inception_model(batch)
                    
                    # 如果features是tuple (在某些inception版本中可能发生)，取第一个元素
                    if isinstance(features, tuple):
                        features = features[0]
                        
                    # 应用softmax
                    if features.ndim == 2:
                        pred = F.softmax(features, dim=-1)
                    else:
                        print(f"Unexpected features dimension: {features.shape}")
                        continue
                        
                    preds_list.append(pred.cpu().numpy())
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

        if not preds_list:
            raise ValueError("No valid predictions were generated")
            
        return np.concatenate(preds_list, axis=0)

    def calculate_inception_score(self, preds, splits=10):
        scores = []
        n_images = preds.shape[0]
        split_size = n_images // splits
        
        # 确保有足够的样本进行分割
        if split_size == 0:
            print("Warning: Too few images for the specified number of splits")
            splits = max(1, n_images // 2)
            split_size = n_images // splits
        
        for i in range(splits):
            part = preds[i * split_size:(i + 1) * split_size]
            kl = part * (np.log(part + 1e-10) - np.log(np.expand_dims(np.mean(part, axis=0) + 1e-10, 0)))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
            
        return np.mean(scores), np.std(scores)

def main():
    parser = argparse.ArgumentParser(description='Calculate Inception Score for iris images')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the iris images')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing (default: 32)')
    parser.add_argument('--splits', type=int, default=10,
                      help='Number of splits for IS calculation (default: 10)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有图像路径
    print(f"Searching for images in {args.data_dir}")
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        pattern = os.path.join(args.data_dir, '**', ext)
        current_paths = glob(pattern, recursive=True)
        image_paths.extend(current_paths)
        print(f"Found {len(current_paths)} {ext} files")
    
    if not image_paths:
        raise ValueError(f"No images found in {args.data_dir}")
    
    print(f"Total images found: {len(image_paths)}")
    
    # 初始化计算器
    calculator = InceptionScoreCalculator(batch_size=args.batch_size)
    
    # 计算激活统计
    print("Calculating activation statistics...")
    preds = calculator.calculate_activation_statistics(image_paths)
    
    # 计算IS评分
    print("Calculating Inception Score...")
    mean, std = calculator.calculate_inception_score(preds, splits=args.splits)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'mean': float(mean),
        'std': float(std),
        'n_images': len(image_paths),
        'batch_size': args.batch_size,
        'splits': args.splits,
        'timestamp': timestamp
    }
    
    # 保存JSON结果
    result_file = os.path.join(args.output_dir, f'is_scores_{timestamp}.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nInception Score: {mean:.3f} ± {std:.3f}")
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()