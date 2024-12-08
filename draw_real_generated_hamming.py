import cv2
import numpy as np
from scipy import signal
import os
from tqdm import tqdm

class StableIrisDetection:
    def detect_iris(self, image_path):
        """稳定的虹膜检测"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return None, "无法读取图像"
                
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 直方图均衡化
            equalized = cv2.equalizeHist(gray)
            
            # 高斯模糊
            blurred = cv2.GaussianBlur(equalized, (7, 7), 0)
            
            # 自适应阈值处理
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # 形态学操作
            kernel = np.ones((3,3), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
            
            # 找到所有轮廓
            contours, _ = cv2.findContours(
                morph,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None, "未检测到轮廓"
            
            # 按面积排序找到最大轮廓
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]
            
            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # 验证圆的合理性
            if radius < 20 or radius > 150:
                return None, "检测到的圆半径不合理"
                
            # 在原图上绘制结果
            result = image.copy()
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            cv2.circle(result, center, 2, (0, 0, 255), 3)
            
            # 提取虹膜区域
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            iris_region = cv2.bitwise_and(gray, gray, mask=mask)
            
            return {
                'center': center,
                'radius': radius,
                'iris_region': iris_region,
                'result_image': result
            }, "成功"
            
        except Exception as e:
            return None, f"处理出错: {str(e)}"
    
    def process_folder(self, input_folder, output_folder):
        """处理整个文件夹的图片"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        results = []
        for img_file in tqdm(image_files):
            input_path = os.path.join(input_folder, img_file)
            result, message = self.detect_iris(input_path)
            
            if result is not None:
                # 保存结果图像
                output_path = os.path.join(output_folder, f"detected_{img_file}")
                cv2.imwrite(output_path, result['result_image'])
                
                # 保存虹膜区域
                iris_path = os.path.join(output_folder, f"iris_{img_file}")
                cv2.imwrite(iris_path, result['iris_region'])
                
                results.append({
                    'file': img_file,
                    'status': 'success',
                    'center': result['center'],
                    'radius': result['radius']
                })
            else:
                results.append({
                    'file': img_file,
                    'status': 'failed',
                    'message': message
                })
                
        return results

def main():
    detector = StableIrisDetection()
    input_folder = "/home/saturn/eyes/realeyes/reference-training"  # 输入文件夹
    output_folder = "/home/saturn/eyes/realeyes/reference-training/results"     # 输出文件夹
    
    print("开始处理图片...")
    results = detector.process_folder(input_folder, output_folder)
    
    # 打印统计信息
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)
    
    print(f"\n处理完成:")
    print(f"总图片数: {total_count}")
    print(f"成功检测: {success_count}")
    print(f"失败数量: {total_count - success_count}")
    
    # 打印失败详情
    print("\n失败详情:")
    for r in results:
        if r['status'] == 'failed':
            print(f"{r['file']}: {r['message']}")

if __name__ == "__main__":
    main()