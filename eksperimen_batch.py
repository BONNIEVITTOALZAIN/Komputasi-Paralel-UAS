import cv2
import time
import numpy as np
import os
import psutil
import glob
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CLASS MONITORING CPU & RAM (Untuk Analisis Mendalam) ---
class SystemMonitor:
    def __init__(self):
        self.keep_running = True
        self.cpu_records = []
        self.thread = None
        
    def _monitor(self):
        # Merekam penggunaan CPU secara berkala setiap 0.5 detik
        while self.keep_running:
            self.cpu_records.append(psutil.cpu_percent(interval=0.5))
            
    def start(self):
        self.keep_running = True
        self.cpu_records = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.keep_running = False
        if self.thread:
            self.thread.join()
        # Mengembalikan rata-rata dari seluruh rekaman CPU
        if len(self.cpu_records) == 0:
            return psutil.cpu_percent(interval=0.1)
        return np.mean(self.cpu_records)

# --- FUNGSI UTAMA PENGOLAHAN ---
def heavy_processing(img):
    cv2.setNumThreads(0) 
    result = img.astype(np.float64)
    for _ in range(50):  
        result = np.sin(result) ** 2 + np.cos(result) ** 2 + result
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def process_single_image(image_path):
    start_task = time.time()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 0
    img = cv2.resize(img, (4000, 4000))
    result = heavy_processing(img)
    duration = time.time() - start_task
    return True, duration
def get_system_memory():
    return psutil.virtual_memory().used / (1024 * 1024)

if __name__ == '__main__':
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Folder '{dataset_dir}' dibuat. Silakan masukkan gambar.")
        exit()

    image_paths = glob.glob(os.path.join(dataset_dir, "*.*"))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [p for p in image_paths if p.lower().endswith(valid_extensions)]
    total_images = len(image_paths)

    if total_images == 0:
        print("Tidak ada gambar di dalam folder dataset.")
        exit()

    print(f"=== EKSPERIMEN BATCH PROCESSING ({total_images} GAMBAR - RESOLUSI 4K) ===\n")
    
    monitor = SystemMonitor()

    # ==========================================
    # 1. PENDEKATAN SERIAL
    # ==========================================
    print("[1] Menjalankan Pendekatan Serial (Memproses 1 per 1)...")
    mem_before_s = get_system_memory()
    monitor.start() # Mulai rekam penggunaan CPU
    start_time_s = time.time()
    
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        _, duration = process_single_image(path)
        print(f"    -> [Serial] Gambar {i+1}/{total_images} ({filename}) selesai dalam {duration:.2f} detik")
        
    end_time_s = time.time()
    avg_cpu_s = monitor.stop() # Hentikan rekaman CPU
    mem_after_s = get_system_memory()
    
    time_serial = end_time_s - start_time_s
    throughput_s = total_images / time_serial

    print("\n--- HASIL METRIK SERIAL ---")
    print(f"Waktu Total          : {time_serial:.4f} detik")
    print(f"Rata-rata per Gambar : {time_serial/total_images:.4f} detik/gambar")
    print(f"Throughput           : {throughput_s * 60:.2f} gambar/menit")
    print(f"Rata-rata CPU Usage  : {avg_cpu_s:.2f} %  <-- (Bukti Serial tidak sanggup memakai semua Core CPU)")
    print(f"Lonjakan RAM         : {abs(mem_after_s - mem_before_s):.2f} MB\n")

    # ==========================================
    # 2. PENDEKATAN PARALEL
    # ==========================================
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    num_cores = logical_cores # Menggunakan seluruh thread yang tersedia

    print(f"[2] Menjalankan Pendekatan Paralel ({num_cores} Thread/Logical Cores)...")
    print(f"    (Info: Laptop Anda memiliki {physical_cores} Physical Cores & {logical_cores} Threads)")
    mem_before_p = get_system_memory()
    monitor.start() # Mulai rekam penggunaan CPU lagi
    start_time_p = time.time()
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Menggunakan as_completed agar ada loading per gambar (Progress Monitor)
        futures = {executor.submit(process_single_image, path): path for path in image_paths}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            path = futures[future]
            filename = os.path.basename(path)
            _, duration = future.result()
            print(f"    -> [Paralel] Gambar {completed}/{total_images} ({filename}) selesai dalam {duration:.2f} detik")
            
    end_time_p = time.time()
    avg_cpu_p = monitor.stop() # Hentikan rekaman CPU
    mem_after_p = get_system_memory()
    
    time_parallel = end_time_p - start_time_p
    throughput_p = total_images / time_parallel

    print("\n--- HASIL METRIK PARALEL ---")
    print(f"Waktu Total          : {time_parallel:.4f} detik")
    print(f"Rata-rata per Gambar : {time_parallel/total_images:.4f} detik/gambar")
    print(f"Throughput           : {throughput_p * 60:.2f} gambar/menit")
    print(f"Rata-rata CPU Usage  : {avg_cpu_p:.2f} %  <-- (Bukti Paralel menggunakan seluruh kapasitas Prosesor)")
    print(f"Lonjakan RAM         : {abs(mem_after_p - mem_before_p):.2f} MB\n")
    
    # ==========================================
    # 3. KESIMPULAN & ANALISIS AKADEMIS
    # ==========================================
    speedup = time_serial / time_parallel
    efisiensi = (speedup / num_cores) * 100
    
    print("=== KESIMPULAN ANALISIS PERFORMA ===")
    print(f"1. Speedup (Percepatan) : {speedup:.2f}x Lebih Cepat")
    print(f"2. Efisiensi Core       : {efisiensi:.2f}% (Berdasarkan rasio Speedup / Jumlah Core)")
    print(f"3. Kapasitas Sistem     : Sistem paralel mampu mengolah {throughput_p/throughput_s:.2f}x lebih banyak gambar dalam waktu yang sama.")
    print("========================================")
