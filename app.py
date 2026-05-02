import cv2
import time
import numpy as np

# Fungsi pemrosesan berat
def heavy_processing(img):
    # Matikan fitur auto-paralel bawaan OpenCV agar komparasi adil
    cv2.setNumThreads(0) 
    
    # Memaksa CPU menghitung operasi sinus & cosinus pada setiap pixel
    result = img.astype(np.float64)
    for _ in range(50):  
        result = np.sin(result) ** 2 + np.cos(result) ** 2 + result
    
    # Normalisasi kembali menjadi gambar standar
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

# Load image (menggunakan gambar 4K Anda)
image = cv2.imread("imresizer-lamborghini-urus-se-3840x2160-26301.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Gambar tidak ditemukan!")
else:
    # Memastikan ukuran gambar cukup besar untuk menyiksa CPU
    if image.shape[0] < 1000:
        image = cv2.resize(image, (4000, 4000))

    print(f"Memulai pemrosesan Serial (Berat) pada resolusi {image.shape}... Harap tunggu.")
    start_time = time.time()

    # Menerapkan pemrosesan berat
    image_processed = heavy_processing(image)

    # Save the processed image
    cv2.imwrite("output_serial.jpg", image_processed)

    end_time = time.time()
    print(f"Serial Processing Time: {end_time - start_time:.4f} seconds")