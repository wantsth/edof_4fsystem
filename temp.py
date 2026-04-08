import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# 生成光瞳孔径和相位掩模
def get_pupil_mask(shape, alpha):
    ny, nx = shape
    y, x = np.ogrid[:ny, :nx]
    y_n, x_n = (y - ny//2)/(ny//2), (x - nx//2)/(nx//2)    # 将坐标归一化到[-1, 1]范围
    r = np.sqrt(x_n**2 + y_n**2)    # 计算每个像素点的径向距离（相对于图像中心）
    aperture = np.where(r < 0.9, 1.0, 0.0)    # 创建一个圆形的光瞳，半径为0.9
    mask = alpha * (x_n**3 + y_n**3)    # 创建一个三次相位掩模
    return aperture, mask

# 模拟图像成像过程，应用离焦相位和相位掩模
def simulate_imaging(image, aperture, mask_phase, z_out, wavelength, f, dx):
    ny, nx = image.shape
    y, x = np.ogrid[:ny, :nx]
    y_n, x_n = (y - ny//2)/(ny//2), (x - nx//2)/(nx//2)
    r_sq = x_n**2 + y_n**2
    L = (nx * dx) / 2    # 像面尺寸（L）
    
    defocus_phase = (np.pi * z_out / (wavelength * f**2)) * r_sq * L**2    # 离焦相位，计算离焦造成的相位差
    pupil = aperture * np.exp(1j * (mask_phase + defocus_phase))    # 计算瞳面复振幅，结合光瞳孔径、相位掩模和离焦相位
    
    psf = np.abs(fftshift(fft2(ifftshift(pupil))))**2    # 计算点扩散函数（PSF）,对瞳面复振幅进行傅里叶变换
    psf /= np.sum(psf)  # 对PSF进行归一化
    psf_f = fft2(ifftshift(psf))    # 对PSF傅里叶变换到频域

    img_f = fft2(image)    # 对输入图像进行傅里叶变换，得到频域图像,经过透镜1
    result = np.real(ifft2(img_f * psf_f))  # 在频域中进行卷积（图像与PSF相乘），然后得到空间域中的成像结果
    
    return np.clip(result, 0, 1), psf    # 将成像结果限制在[0, 1]范围内

# 维纳滤波复原：基于维纳滤波对模糊图像进行复原
def wiener_deconv(image, psf_ref, K=0.01):
    # 对输入图像进行傅里叶变换
    img_f = fft2(image)
    # 对参考PSF进行傅里叶变换
    psf_f = fft2(ifftshift(psf_ref))
    # 计算PSF的共轭复数
    H_conj = np.conj(psf_f)
    # 计算PSF的能量（功率谱）
    H_power = np.abs(psf_f)**2
    # 计算维纳滤波器： H* / (|H|^2 + K)
    wiener_filter = H_conj / (H_power + K)
    # 在频域中进行滤波，最后通过反傅里叶变换得到复原图像
    restored = np.real(ifft2(img_f * wiener_filter))
    # 将复原结果限制在[0, 1]范围内
    return np.clip(restored, 0, 1)

if __name__ == "__main__":
    wavelength = 532e-9  # 波长 m 
    f = 0.2              # 焦距 m
    dx = 10e-6           # 像素尺寸 m
    pad_size = 5         # 填充大小
    alpha = 25.0         # 相位掩模强度
    img = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # 将图像调整为 256x256
    img = img.astype(np.float32) / 255.0  # 归一化图像到 [0, 1]
  
    img_pad = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='edge')  # 边缘填充（减少边界伪影）

    ap_norm, phase_norm = get_pupil_mask(img_pad.shape, 0)        # 普通系统
    ap_wavefront, phase_wavefront = get_pupil_mask(img_pad.shape, alpha)      # 波前编码系统

    _, psf_wf = simulate_imaging(img_pad, ap_wavefront, phase_wavefront, 0, wavelength, f, dx)    # 获取参考PSF（零离焦）

    z_vals = np.linspace(-f, f, 41)  # 离焦量范围
    for z_out in z_vals:
        # 普通系统成像
        img_normal, _ = simulate_imaging(img_pad, ap_norm, phase_norm, z_out, wavelength, f, dx)
        # 波前编码系统成像（编码模糊）
        img_wf_encoded, _ = simulate_imaging(img_pad, ap_wavefront, phase_wavefront, z_out, wavelength, f, dx)
        # 波前编码复原（使用维纳滤波）
        img_wf_restored = wiener_deconv(img_wf_encoded, psf_wf)
        plt.subplot(1, 2, 1)
        plt.title(f"Output at z={z_out:.3f} m")
        plt.imshow(img_normal, cmap='gray')
        plt.axis('off')
        #plt.savefig(f"normal/z{z_out:.3f}.png", dpi=150, bbox_inches='tight')
        
        plt.subplot(1, 2, 2)
        plt.title("Wiener Deconvolution")
        plt.imshow(img_wf_restored, cmap='gray')
        plt.axis('off')
        #plt.savefig(f"restored/z{z_out:.3f}.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"compare/z{z_out:.3f}.png", dpi=150, bbox_inches='tight')
        plt.pause(0.25)  # 暂停0.5秒