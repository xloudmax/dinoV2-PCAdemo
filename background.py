import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA

# 常量定义
PATCH_H, PATCH_W = 120, 120  # 每个图像块的高度和宽度，减小以优化显存使用
FEAT_DIM = 1536  # ViT-G/14 模型特征向量的维度

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 如果有 CUDA 可用则使用 GPU，否则使用 CPU

# 图像预处理函数
def preprocess_image(img_path):
    """
    加载并对图像进行预处理，包括高斯模糊、缩放、裁剪、标准化等操作。
    """
    try:
        img = Image.open(img_path).convert('RGB')  # 打开图像并转换为 RGB 模式
    except Exception as e:
        print(f"无法打开图像 {img_path}，错误信息: {e}")
        return None

    # 图像预处理的操作序列
    transform = T.Compose([
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),  # 应用高斯模糊
        T.Resize((PATCH_H * 14, PATCH_W * 14)),  # 缩放图像
        T.CenterCrop((PATCH_H * 14, PATCH_W * 14)),  # 中心裁剪
        T.ToTensor(),  # 转换为张量
        T.Normalize(mean=(0.485, 0.456, 0.406),  # 标准化
                    std=(0.229, 0.224, 0.225)),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)  # 增加批量维度并将张量移至设备
    return img_tensor

# 使用 scikit-learn 的 PCA 函数
def pca(features, n_components=3):
    """
    对特征进行 PCA 降维操作。
    """
    pca = PCA(n_components=n_components)  # 创建 PCA 对象
    pca_features = pca.fit_transform(features)  # 降维
    return pca_features
def pca_sklearn(features, n_components=3):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    return pca_features
# 特征提取函数
def extract_features(model, img_tensor):
    """
    从模型中提取特征，使用 autocast 提高效率。
    """
    with torch.no_grad():  # 禁用梯度计算以节省显存
        with torch.amp.autocast('cuda'):  # 使用混合精度，指定设备为 'cuda'
            features_dict = model.forward_features(img_tensor)  # 提取特征
            features = features_dict['x_norm_patchtokens'].reshape(-1, FEAT_DIM)  # 重塑为 (N, FEAT_DIM)
    return features


# PCA 降维和前景/背景分离
def process_with_pca(features, n_components=3, fg_threshold=0.3):
    """
    对特征执行 PCA 降维，并根据主成分分离前景和背景。
    """
    features_cpu = features.cpu().numpy()  # 将张量移到 CPU 并转换为 NumPy 数组

    # 对所有特征执行 PCA
    pca_features = pca_sklearn(features_cpu, n_components)

    # 使用第一主成分分离前景和背景
    fg_mask = pca_features[:, 0] > fg_threshold  # 前景条件
    bg_mask = ~fg_mask  # 背景为条件的补集

    # 如果有前景，则对前景特征再次进行 PCA
    if np.any(fg_mask):
        fg_features = features_cpu[fg_mask]
        pca_features_fg_norm = pca_sklearn(fg_features, n_components)
        # 归一化前景特征到 [0, 1]
        pca_features_fg_norm = (pca_features_fg_norm - pca_features_fg_norm.min(axis=0)) / \
                               (pca_features_fg_norm.max(axis=0) - pca_features_fg_norm.min(axis=0) + 1e-8)
    else:
        pca_features_fg_norm = np.zeros((0, n_components))

    return pca_features, fg_mask, bg_mask, pca_features_fg_norm

# 可视化 PCA 特征
def visualize_pca(pca_features_rgb):
    """
    将 PCA 特征可视化为 RGB 图像。
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(pca_features_rgb.astype(np.float32))
    plt.axis('off')
    plt.show()


# 主函数
def main(img_path, model_path='/home/xloudmax/桌面/dinov2'):
    start_time = time.time()

    # 清理显存缓存
    torch.cuda.empty_cache()

    # 加载模型
    try:
        model = torch.hub.load(model_path, 'dinov2_vitg14', source='local').to(device)
        model.eval()  # 设置为评估模式
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 图像预处理
    img_tensor = preprocess_image(img_path)
    if img_tensor is None:
        return

    # 特征提取
    features = extract_features(model, img_tensor)

    # 如果模型不再需要，释放显存
    del model
    torch.cuda.empty_cache()

    # PCA 降维和前景/背景分离
    pca_features, pca_features_fg, pca_features_bg, pca_features_rem = process_with_pca(features, n_components=3,
                                                                                        fg_threshold=0.3)

    # 生成 RGB 可视化
    pca_features_rgb = pca_features.copy()
    pca_features_rgb[pca_features_fg] = pca_features_rem  # 替换为处理后的前景特征
    pca_features_rgb[pca_features_bg] = 0  # 背景置为 0

    # 重塑为图像形状
    try:
        pca_features_rgb = pca_features_rgb.reshape(PATCH_H, PATCH_W, 3)  # 形状为 (150, 150, 3)
    except ValueError as e:
        print(f"重塑 PCA 特征时出错: {e}")
        return

    # 归一化 RGB 图像到 [0,1]
    pca_features_rgb = pca_features_rgb - pca_features_rgb.min()
    pca_features_rgb = pca_features_rgb / (pca_features_rgb.max() + 1e-8)

    # 可视化
    visualize_pca(pca_features_rgb)

    # 打印总耗时
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.4f} 秒")


# 脚本入口
if __name__ == "__main__":
    img_path = r'/home/xloudmax/桌面/dinov2/OIP-C (1).jpeg'  # 输入图像路径
    main(img_path)
