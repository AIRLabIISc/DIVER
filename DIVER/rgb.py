import os
import csv
from PIL import Image
import numpy as np
import cv2

# --- Function to get average RGB and HSV of an image ---
def get_avg_rgb_hsv_values(image_path):
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        pixels_rgb = np.array(img_rgb)

        # Average RGB
        avg_rgb = pixels_rgb.mean(axis=(0, 1))

        # Convert to HSV using OpenCV
        bgr_img = cv2.cvtColor(pixels_rgb, cv2.COLOR_RGB2BGR)
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        avg_hsv = hsv_img.mean(axis=(0, 1))

        return avg_rgb, avg_hsv


# --- Process images and save CSV ---
def process_images_and_save_csv(folder_path, csv_save_path, append_final_avg=True):
    all_values = []
    rgb_sums = np.zeros(3)
    hsv_sums = np.zeros(3)
    count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            image_path = os.path.join(folder_path, filename)
            try:
                avg_rgb, avg_hsv = get_avg_rgb_hsv_values(image_path)

                rgb_sums += avg_rgb
                hsv_sums += avg_hsv
                count += 1

                all_values.append([
                    filename,
                    avg_rgb[0], avg_rgb[1], avg_rgb[2],
                    avg_hsv[0], avg_hsv[1], avg_hsv[2]
                ])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    # ---------- FIX STARTS HERE ----------
    csv_dir = os.path.dirname(csv_save_path)
    if csv_dir:  # handles case when path is just "file.csv"
        os.makedirs(csv_dir, exist_ok=True)
    # ---------- FIX ENDS HERE ----------

    # Save results to CSV
    with open(csv_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Image Name", "Red", "Green", "Blue", "Hue", "Saturation", "Value"]
        )
        writer.writerows(all_values)

    print(f"\nPer-image RGB and HSV values saved to: {csv_save_path}")

    # Compute and print overall averages
    if count > 0:
        final_avg_rgb = rgb_sums / count
        final_avg_hsv = hsv_sums / count

        print("\n===== FINAL AVERAGE (ALL IMAGES) =====")
        print(f"Average RGB: R={final_avg_rgb[0]:.2f}, "
              f"G={final_avg_rgb[1]:.2f}, "
              f"B={final_avg_rgb[2]:.2f}")
        print(f"Average HSV: H={final_avg_hsv[0]:.2f}, "
              f"S={final_avg_hsv[1]:.2f}, "
              f"V={final_avg_hsv[2]:.2f}")

        if append_final_avg:
            with open(csv_save_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "FINAL_AVERAGE",
                    final_avg_rgb[0], final_avg_rgb[1], final_avg_rgb[2],
                    final_avg_hsv[0], final_avg_hsv[1], final_avg_hsv[2]
                ])
            print("Final averages appended to CSV.")
    else:
        print("No valid images found in the folder.")
