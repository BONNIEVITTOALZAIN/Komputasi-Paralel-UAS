import cv2
import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# 1. Fungsi Pemrosesan Berat (Serial & Paralel memiliki beban komputasi persis sama)
def heavy_processing(img):
    cv2.setNumThreads(0) 
    result = img.astype(np.float64)
    for _ in range(50):  
        result = np.sin(result) ** 2 + np.cos(result) ** 2 + result
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def process_block(start_row, end_row, image):
    cv2.setNumThreads(0) 
    block = image[start_row:end_row, :].astype(np.float64)
    for _ in range(50):
        block = np.sin(block) ** 2 + np.cos(block) ** 2 + block
    block = cv2.normalize(block, None, 0, 255, cv2.NORM_MINMAX)
    return block.astype(np.uint8)

# Fungsi untuk memonitor memori RAM
def get_system_memory():
    """Mengukur total RAM yang digunakan oleh sistem operasi saat ini (dalam MB)"""
    return psutil.virtual_memory().used / (1024 * 1024)

if __name__ == '__main__':
    # Membaca gambar asli
    image_path = "imresizer-lamborghini-urus-se-3840x2160-26301.jpg"
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        print(f"Error: Gambar '{image_path}' tidak ditemukan!")
        exit()

    # 1. UKURAN DATASET YANG BERAGAM (Uji Skalabilitas)
    # Kita menguji resolusi kecil, menengah, dan sangat besar
    sizes = [1000, 2000, 3000, 4000]
    
    serial_times = []
    parallel_times = []
    
    serial_throughputs = []
    parallel_throughputs = []

    print("=== MULAI EKSPERIMEN PERBANDINGAN PERFORMA ===")

    for size in sizes:
        print(f"\n--- Menguji Resolusi {size}x{size} ---")
        test_image = cv2.resize(original_image, (size, size))
        total_pixels = size * size

        # --- PENDEKATAN SERIAL ---
        print("Menjalankan Pendekatan Serial...")
        mem_before = get_system_memory()
        start_time = time.time()
        
        img_serial = heavy_processing(test_image)
        
        end_time = time.time()
        mem_after = get_system_memory()
        
        time_serial = end_time - start_time
        serial_times.append(time_serial)
        
        # 3. MENGUKUR THROUGHPUT (Berapa juta piksel yang diolah per detik)
        throughput_s = total_pixels / time_serial 
        serial_throughputs.append(throughput_s)
        
        print(f"Waktu Serial   : {time_serial:.4f} detik")
        print(f"Lonjakan RAM   : {abs(mem_after - mem_before):.2f} MB")
        print(f"Throughput     : {throughput_s / 1e6:.2f} Juta Piksel/detik")

        # --- PENDEKATAN PARALEL ---
        print("Menjalankan Pendekatan Paralel (4 Core)...")
        num_cores = 4
        rows_per_core = test_image.shape[0] // num_cores
        
        mem_before = get_system_memory()
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i in range(num_cores):
                start_row = i * rows_per_core
                end_row = (i + 1) * rows_per_core if i != num_cores - 1 else test_image.shape[0]
                futures.append(executor.submit(process_block, start_row, end_row, test_image))
            processed_blocks = [future.result() for future in futures]
            
        img_parallel = cv2.vconcat(processed_blocks)
        
        end_time = time.time()
        mem_after = get_system_memory()
        
        time_parallel = end_time - start_time
        parallel_times.append(time_parallel)
        throughput_p = total_pixels / time_parallel
        parallel_throughputs.append(throughput_p)
        
        print(f"Waktu Paralel  : {time_parallel:.4f} detik")
        print(f"Lonjakan RAM   : {abs(mem_after - mem_before):.2f} MB")
        print(f"Throughput     : {throughput_p / 1e6:.2f} Juta Piksel/detik")

        # Simpan contoh gambar hasil untuk visualisasi
        if size == sizes[-1]:
            cv2.imwrite("hasil_serial_visual.jpg", img_serial)
            cv2.imwrite("hasil_paralel_visual.jpg", img_parallel)

    # --- 4. VISUALISASI HASIL DENGAN MATPLOTLIB ---
    print("\nMenyiapkan Grafik Visualisasi...")
    
    plt.figure(figsize=(12, 5))

    # Grafik 1: Waktu Eksekusi vs Ukuran Gambar
    plt.subplot(1, 2, 1)
    plt.plot(sizes, serial_times, marker='o', label='Serial', color='red')
    plt.plot(sizes, parallel_times, marker='o', label='Paralel (4 Core)', color='blue')
    plt.title('Perbandingan Waktu Eksekusi')
    plt.xlabel('Ukuran Gambar (Pixel x Pixel)')
    plt.ylabel('Waktu Eksekusi (Detik) - Makin kecil makin bagus')
    plt.grid(True)
    plt.legend()

    # Grafik 2: Throughput vs Ukuran Gambar
    plt.subplot(1, 2, 2)
    # Konversi throughput ke satuan Juta Piksel per Detik (Megapixels/s)
    s_throughputs_mp = [t / 1e6 for t in serial_throughputs]
    p_throughputs_mp = [t / 1e6 for t in parallel_throughputs]
    
    plt.plot(sizes, s_throughputs_mp, marker='s', label='Serial', color='red')
    plt.plot(sizes, p_throughputs_mp, marker='s', label='Paralel (4 Core)', color='blue')
    plt.title('Perbandingan Throughput (Kapasitas Proses)')
    plt.xlabel('Ukuran Gambar (Pixel x Pixel)')
    plt.ylabel('Throughput (Juta Piksel/Detik) - Makin besar makin bagus')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('grafik_perbandingan.png')
    plt.show()
    
    print("Selesai! Grafik telah disimpan sebagai 'grafik_perbandingan.png'.")
    print("Gambar hasil serial dan paralel juga telah disimpan untuk bukti visualisasi.")
