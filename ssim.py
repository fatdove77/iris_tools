import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import json
from datetime import datetime

class ResolutionAwareSSIMCalculator:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 设置基础路径
        self.real_dir = "/home/saturn/eyes/realeyes/all"
        self.base_gen_dir = "/home/saturn/eyes/1024_sampling_classified"
        
        # 创建结果目录
        self.results_dir = "ssim_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 预处理设置
        self.real_preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.gen_preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def create_gaussian_kernel(self, window_size=11, sigma=1.5, channels=3):
        x_coord = torch.arange(window_size, dtype=torch.float32)
        x_grid = x_coord.repeat(window_size).view(window_size, window_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (window_size - 1) / 2.
        variance = sigma ** 2.

        gaussian_kernel = torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )

        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, window_size, window_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        return gaussian_kernel

    def ssim(self, img1, img2, window_size=11):
        try:
            if img1.shape != img2.shape:
                print(f"Shape mismatch - img1: {img1.shape}, img2: {img2.shape}")
                raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

            window = self.create_gaussian_kernel(
                window_size=window_size, 
                channels=img1.size(1)
            ).to(img1.device)

            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=3)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=3)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=3) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=3) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=3) - mu1_mu2

            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            return ssim_map.mean()

        except Exception as e:
            print(f"Error in SSIM calculation: {str(e)}")
            return None

    def load_real_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = self.real_preprocess(img)
            return img_tensor
        except Exception as e:
            print(f"Error loading real image {path}: {str(e)}")
            return None

    def load_gen_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = self.gen_preprocess(img)
            return img_tensor
        except Exception as e:
            print(f"Error loading generated image {path}: {str(e)}")
            return None

    def calculate_dataset_ssim(self, gen_dir):
        gen_files = [f for f in os.listdir(gen_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        real_files = [f for f in os.listdir(self.real_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(gen_files)} generated images from {gen_dir}...")
        
        # 加载真实图像
        real_tensors = []
        valid_real_files = []
        for real_file in tqdm(real_files, desc="Loading real images"):
            tensor = self.load_real_image(os.path.join(self.real_dir, real_file))
            if tensor is not None:
                if tensor.shape[-2:] != (256, 256):
                    print(f"Warning: Real image {real_file} has unexpected shape: {tensor.shape}")
                    continue
                real_tensors.append(tensor)
                valid_real_files.append(real_file)
        
        if not real_tensors:
            raise ValueError("No valid real images loaded")
            
        real_tensors = torch.stack(real_tensors).to(self.device)
        
        results = []
        for gen_file in tqdm(gen_files, desc="Processing generated images"):
            gen_tensor = self.load_gen_image(os.path.join(gen_dir, gen_file))
            if gen_tensor is not None:
                if gen_tensor.shape[-2:] != (256, 256):
                    print(f"Warning: Generated image {gen_file} has unexpected shape: {gen_tensor.shape}")
                    continue
                    
                gen_tensor = gen_tensor.to(self.device)
                
                ssim_scores = []
                for real_tensor in real_tensors:
                    score = self.ssim(gen_tensor.unsqueeze(0), real_tensor.unsqueeze(0))
                    if score is not None:
                        ssim_scores.append(score.item())
                
                if ssim_scores:
                    best_idx = np.argmax(ssim_scores)
                    results.append({
                        'generated_file': gen_file,
                        'best_match_file': valid_real_files[best_idx],
                        'ssim_score': ssim_scores[best_idx]
                    })
        
        return results

    def run_evaluation(self):
        print("Starting evaluation for multiple folders...")
        all_results = {}
        
        # 遍历从50000到225000，步长为25000的所有文件夹
        for step in range(75000, 200001, 25000):
            gen_dir = os.path.join(self.base_gen_dir, str(step), "images")
            
            if not os.path.exists(gen_dir):
                print(f"Warning: Directory {gen_dir} does not exist, skipping...")
                continue
                
            print(f"\nProcessing directory: {gen_dir}")
            results = self.calculate_dataset_ssim(gen_dir)
            
            if not results:
                print(f"No valid results obtained for {gen_dir}")
                continue
            
            ssim_scores = [r['ssim_score'] for r in results]
            stats = {
                'mean': float(np.mean(ssim_scores)),
                'std': float(np.std(ssim_scores)),
                'min': float(np.min(ssim_scores)),
                'max': float(np.max(ssim_scores)),
                'median': float(np.median(ssim_scores)),
                'num_comparisons': len(results)
            }
            
            all_results[str(step)] = stats
            
            print(f"\nResults for step {step}:")
            print(f"Mean SSIM: {stats['mean']:.4f} ± {stats['std']:.4f}")
            # print(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            # print(f"Median: {stats['median']:.4f}")
            # print(f"Number of comparisons: {stats['num_comparisons']}")
        
        # 保存所有结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'ssim_results_all_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nAll results saved to: {results_file}")

def main():
    calculator = ResolutionAwareSSIMCalculator(batch_size=32)
    calculator.run_evaluation()

if __name__ == '__main__':
    main()