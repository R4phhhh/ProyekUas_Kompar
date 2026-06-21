import os
import time
import cv2
import cupy as cp
import numpy as np
from multiprocessing import Pool

def enhance_captcha(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return False

    _, enhanced = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    enhanced = cv2.medianBlur(enhanced, 3)
    return True

def enhance_captcha_gpu(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return False

    gpu_img = cp.asarray(img)
    threshold = cp.mean(gpu_img)

    enhanced = cp.where(
        gpu_img > threshold,
        255,
        0
    ).astype(cp.uint8)

    cp.cuda.Stream.null.synchronize()
    cp.asnumpy(enhanced)
    return True

if __name__ == "__main__":
    dataset_dir = "./dataset_10000"

    if not os.path.exists(dataset_dir):
        print(f"Error: Folder '{dataset_dir}' tidak ditemukan.")
        exit()

    image_paths = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    total_images = len(image_paths)

    if total_images == 0:
        print(f"Tidak ada gambar di folder '{dataset_dir}'.")
        exit()

    print(f"=== Pengujian {total_images} CAPTCHA ===")
    print("\nGPU Information")

    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"GPU Device Count : {gpu_count}")
        if gpu_count > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props["name"]

            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()

            print(f"GPU Name : {gpu_name}")

        gpu_available = gpu_count > 0

    except Exception as e:
        gpu_available = False

        print("GPU tidak tersedia")
        print(e)

    start_seq = time.perf_counter()

    for path in image_paths:
        enhance_captcha(path)

    end_seq = time.perf_counter()

    time_seq = end_seq - start_seq

    print(f"\nSequential CPU : {time_seq:.4f} detik")

    num_processors = os.cpu_count()

    start_para = time.perf_counter()

    with Pool(processes=num_processors) as pool:
        pool.map(enhance_captcha, image_paths)

    end_para = time.perf_counter()

    time_para = end_para - start_para

    print(
        f"Multiprocessing CPU ({num_processors} core) : "
        f"{time_para:.4f} detik"
    )

    if gpu_available:
        start_gpu = time.perf_counter()
        for path in image_paths:
            enhance_captcha_gpu(path)

        cp.cuda.Stream.null.synchronize()
        end_gpu = time.perf_counter()
        time_gpu = end_gpu - start_gpu 
        print(f"GPU CuPy : {time_gpu:.4f} detik")
    else:
        time_gpu = None

    speedup_cpu = time_seq / time_para
    efficiency_cpu = speedup_cpu / num_processors

    throughput_seq = total_images / time_seq
    throughput_para = total_images / time_para

    if time_gpu is not None:
        throughput_gpu = total_images / time_gpu
        speedup_gpu = time_seq / time_gpu

    print("\n" + "=" * 70)
    print("        CAPTCHA IMAGE ENHANCEMENT PERFORMANCE ANALYSIS")
    print("=" * 70)

    print(f"Total Dataset      : {total_images:,} images")
    print(f"CPU Logical Cores  : {num_processors}")

    if gpu_available:
        print(f"GPU Device         : {gpu_name}")

    print("-" * 70)

    print(
        f"{'Method':<20}"
        f"{'Time (s)':>15}"
        f"{'Throughput':>20}"
        f"{'Speedup':>15}"
    )

    print("-" * 70)

    print(
        f"{'Sequential CPU':<20}"
        f"{time_seq:>15.4f}"
        f"{throughput_seq:>20.2f}"
        f"{'1.00x':>15}"
    )

    print(
        f"{'Parallel CPU':<20}"
        f"{time_para:>15.4f}"
        f"{throughput_para:>20.2f}"
        f"{speedup_cpu:>15.2f}x"
    )

    if time_gpu is not None:
        print(
            f"{'GPU CuPy':<20}"
            f"{time_gpu:>15.4f}"
            f"{throughput_gpu:>20.2f}"
            f"{speedup_gpu:>15.2f}x"
        )

    print("-" * 70)

    print(
        f"CPU Efficiency     : "
        f"{efficiency_cpu:.2f} ({efficiency_cpu*100:.2f}%)"
    )

    print("=" * 70)