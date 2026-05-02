import cv2
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Function for processing a block of the image
def process_block(start_row, end_row, image):
    # Matikan auto-paralel OpenCV agar adil, karena kita menguji proses Python-nya
    cv2.setNumThreads(0) 
    
    block = image[start_row:end_row, :].astype(np.float64)
    # Perhitungan rumit yang sama persis
    for _ in range(50):
        block = np.sin(block) ** 2 + np.cos(block) ** 2 + block
        
    block = cv2.normalize(block, None, 0, 255, cv2.NORM_MINMAX)
    return block.astype(np.uint8)

if __name__ == '__main__':
    # Load image (menggunakan gambar 4K Anda)
    image = cv2.imread("imresizer-lamborghini-urus-se-3840x2160-26301.jpg", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Gambar tidak ditemukan!")
    else:
        # Memastikan ukuran gambar cukup besar
        if image.shape[0] < 1000:
            image = cv2.resize(image, (4000, 4000))
            
        print(f"Memulai pemrosesan Paralel (Berat) pada resolusi {image.shape}... Harap tunggu.")
        # Define block size based on number of CPU cores
        num_cores = 4  # Example: Using 4 cores
        rows_per_core = image.shape[0] // num_cores

        start_time = time.time()

        # Use ProcessPoolExecutor to parallelize image processing
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i in range(num_cores):
                start_row = i * rows_per_core
                end_row = (i + 1) * rows_per_core if i != num_cores - 1 else image.shape[0]
                futures.append(executor.submit(process_block, start_row, end_row, image))

            # Collect the results
            processed_blocks = [future.result() for future in futures]

        # Reassemble the image
        image_processed = cv2.vconcat(processed_blocks)

        # Save the processed image
        cv2.imwrite("output_parallel.jpg", image_processed)

        end_time = time.time()
        print(f"Parallel Processing Time: {end_time - start_time:.4f} seconds")