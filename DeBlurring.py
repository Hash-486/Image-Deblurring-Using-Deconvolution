import numpy as np
import cv2
from skimage import img_as_float
from skimage.restoration import richardson_lucy, wiener, denoise_bilateral, denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def apply_laplacian(image):
    """Apply Laplacian filter for deblurring."""
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return convolve2d(image, kernel, mode='same', boundary='wrap')

def ideal_high_pass_filter(image, cutoff=30):
    """Apply ideal high-pass filter in frequency domain."""
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.ogrid[0:rows, 0:cols]
    mask = np.sqrt((x - crow)**2 + (y - ccol)**2) > cutoff
    dft_shifted_filtered = dft_shifted * mask

    return np.fft.ifft2(np.fft.ifftshift(dft_shifted_filtered)).real

def apply_filters(image):
    """Apply Gaussian, Bilateral, and TV denoising filters."""
    gaussian_denoised = gaussian_filter(image, sigma=1)  # Gaussian denoising
    bilateral_denoised = denoise_bilateral(image, sigma_color=0.1, sigma_spatial=15)
    tv_denoised = denoise_tv_chambolle(image, weight=0.1)
    return gaussian_denoised, bilateral_denoised, tv_denoised

def sharpen_image(image, alpha=1.5):
    """Apply sharpening to the image."""
    blurred = gaussian_filter(image, sigma=1)
    return np.clip(image + alpha * (image - blurred), 0, 1)

def blind_deconvolution(blurred_image, psf, num_iter=30):
    """Perform blind deconvolution using Richardson-Lucy."""
    blurred_image = img_as_float(blurred_image)
    return richardson_lucy(blurred_image, psf, num_iter=num_iter)

def wiener_deconvolution(blurred_image, psf, balance=0.1):
    """Perform Wiener deconvolution."""
    blurred_image = img_as_float(blurred_image)
    return wiener(blurred_image, psf, balance=balance)

def calculate_metrics(original, processed):
    """Calculate accuracy metrics: SSIM and PSNR."""
    ssim_value = ssim(original, processed, data_range=processed.max() - processed.min())
    psnr_value = psnr(original, processed, data_range=processed.max() - processed.min())
    return ssim_value, psnr_value

def select_image_file():
    """Open a dialog to select an image file."""
    Tk().withdraw()  # Close the root window
    return askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

def main():
    # Load and preprocess the image
    image_path = select_image_file()  # Use the file dialog to select the image
    if not image_path:  # Check if a file was selected
        print("No file selected. Exiting...")
        return
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = img_as_float(original_image)

    # Define PSF (motion blur kernel)
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel /= kernel.sum()

    # Generate blurred image
    blurred_image = cv2.filter2D(original_image, -1, kernel)

    # Apply Laplacian deblurring
    laplacian_deblurred = apply_laplacian(original_image)
    laplacian_ssim, laplacian_psnr = calculate_metrics(original_image, laplacian_deblurred)

    # Apply Ideal High-Pass Filtering
    high_pass_deblurred = ideal_high_pass_filter(original_image, cutoff=30)
    hp_ssim, hp_psnr = calculate_metrics(original_image, high_pass_deblurred)

    # Richardson-Lucy Deblurring
    deblurred_rl = blind_deconvolution(blurred_image, kernel)
    accuracy_rl = calculate_metrics(original_image, deblurred_rl)

    # Wiener Deblurring
    deblurred_wiener = wiener_deconvolution(blurred_image, kernel, balance=0.1)
    accuracy_wiener = calculate_metrics(original_image, deblurred_wiener)

    # Apply Denoising Filters
    gaussian_denoised, bilateral_denoised, tv_denoised = apply_filters(high_pass_deblurred)

    # Sharpen the final images
    sharpened_gaussian = sharpen_image(gaussian_denoised)
    sharpened_bilateral = sharpen_image(bilateral_denoised)
    sharpened_tv = sharpen_image(tv_denoised)

    # Calculate metrics for the denoised images
    gaussian_ssim, gaussian_psnr = calculate_metrics(original_image, sharpened_gaussian)
    bilateral_ssim, bilateral_psnr = calculate_metrics(original_image, sharpened_bilateral)
    tv_ssim, tv_psnr = calculate_metrics(original_image, sharpened_tv)

    # Display results
    plt.figure(figsize=(15, 15))
    
    # Original Image
    plt.subplot(3, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')

    # Laplacian Deblurring
    plt.subplot(3, 3, 2)
    plt.title(f'Laplacian Deblurring\nSSIM: {laplacian_ssim:.2f}, PSNR: {laplacian_psnr:.2f}')
    plt.imshow(laplacian_deblurred, cmap='gray')

    # Ideal High-Pass Deblurring
    plt.subplot(3, 3, 3)
    plt.title(f'Ideal High-Pass Deblurring\nSSIM: {hp_ssim:.2f}, PSNR: {hp_psnr:.2f}')
    plt.imshow(high_pass_deblurred, cmap='gray')

    # Gaussian Denoising
    plt.subplot(3, 3, 4)
    plt.title(f'Gaussian Denoising\nSSIM: {gaussian_ssim:.2f}, PSNR: {gaussian_psnr:.2f}')
    plt.imshow(sharpened_gaussian, cmap='gray')

    # Bilateral Denoising
    plt.subplot(3, 3, 5)
    plt.title(f'Bilateral Denoising\nSSIM: {bilateral_ssim:.2f}, PSNR: {bilateral_psnr:.2f}')
    plt.imshow(sharpened_bilateral, cmap='gray')

    # TV Denoising
    plt.subplot(3, 3, 6)
    plt.title(f'TV Denoising\nSSIM: {tv_ssim:.2f}, PSNR: {tv_psnr:.2f}')
    plt.imshow(sharpened_tv, cmap='gray')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
