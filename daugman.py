import numpy as np
import cv2
from scipy import signal
import math
import os
from itertools import combinations
from tqdm import tqdm

class IrisRecognition:
    def __init__(self):
        self.iris_radius = None
        self.pupil_radius = None
        self.center = None
        
    def segment_iris(self, eye_image):
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            self.center = (circles[0,0,0], circles[0,0,1])
            self.pupil_radius = circles[0,0,2]
            
            outer_circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=self.pupil_radius + 20,
                maxRadius=self.pupil_radius + 100
            )
            
            if outer_circles is not None:
                outer_circles = np.uint16(np.around(outer_circles))
                self.iris_radius = outer_circles[0,0,2]
                return True
        return False
    
    def to_polar(self, image):
        polar_array = np.zeros((100, 360))
        center_x, center_y = self.center
        
        for r in range(100):
            for theta in range(360):
                r_val = self.pupil_radius + (r/100.0)*(self.iris_radius - self.pupil_radius)
                x = int(center_x + r_val * math.cos(theta * math.pi / 180))
                y = int(center_y + r_val * math.sin(theta * math.pi / 180))
                
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    polar_array[r, theta] = image[y, x]
                    
        return polar_array
    
    def extract_features(self, polar_iris):
        features = []
        sigma = 3.0
        freq = 0.1
        
        for theta in [0, 45, 90, 135]:
            theta_rad = theta * np.pi / 180
            kernel_size = 31
            kernel = np.zeros((kernel_size, kernel_size))
            
            for x in range(-15, 16):
                for y in range(-15, 16):
                    x_theta = x * np.cos(theta_rad) + y * np.sin(theta_rad)
                    y_theta = -x * np.sin(theta_rad) + y * np.cos(theta_rad)
                    kernel[x+15, y+15] = np.exp(-.5*(x_theta**2 + y_theta**2)/(sigma**2)) * \
                                       np.cos(2*np.pi*freq*x_theta)
            
            filtered = signal.convolve2d(polar_iris, kernel, mode='same')
            features.extend(filtered.flatten() > 0)
            
        return np.array(features)
    
    def process_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                return None
                
            if not self.segment_iris(image):
                print(f"无法检测到虹膜: {image_path}")
                return None
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            polar = self.to_polar(gray)
            features = self.extract_features(polar)
            
            return features
            
        except Exception as e:
            print(f"处理图片时出错 {image_path}: {str(e)}")
            return None
    
    def hamming_distance(self, template1, template2):
        return np.sum(template1 != template2) / len(template1)
    
    def process_folder(self, folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        n = len(image_files)
        results = np.zeros((n, n), dtype=float)
        features_dict = {}
        
        print("正在提取特征...")
        for i, img_file in enumerate(tqdm(image_files)):
            img_path = os.path.join(folder_path, img_file)
            features = self.process_image(img_path)
            if features is not None:
                features_dict[img_file] = features
        
        print("正在计算Hamming距离...")
        pairs_results = []
        for (img1, feat1), (img2, feat2) in tqdm(list(combinations(features_dict.items(), 2))):
            distance = self.hamming_distance(feat1, feat2)
            i = image_files.index(img1)
            j = image_files.index(img2)
            results[i,j] = distance
            results[j,i] = distance
            
            if distance < 0.32:
                pairs_results.append((img1, img2, distance))
                
        return results, pairs_results, image_files

def main():
    recognizer = IrisRecognition()
    folder_path = "/home/saturn/eyes/realeyes/reference-training"  # 请替换为您的图片文件夹路径
    
    distance_matrix, matching_pairs, image_files = recognizer.process_folder(folder_path)
    
    print("\n可能来自同一个体的图片对:")
    for img1, img2, distance in sorted(matching_pairs, key=lambda x: x[2]):
        print(f"{img1} - {img2}: Hamming距离 = {distance:.4f}")
    
    np.savetxt("distances.csv", distance_matrix, delimiter=",")
    with open("image_files.txt", "w") as f:
        f.write("\n".join(image_files))
    
    print(f"\n共处理了 {len(image_files)} 张图片")
    print(f"距离矩阵已保存到 distances.csv")
    print(f"图片文件列表已保存到 image_files.txt")

if __name__ == "__main__":
    main()