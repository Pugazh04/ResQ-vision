import cv2
import os
import time

rtsp_url = "<your 5G camera url>" # IP
output_folder = "camera_input"
os.makedirs(output_folder, exist_ok=True)
target_width = 384
target_height = 384

SAVE_EVERY_N_FRAMES = 1

def capture_img():
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Error: Could not open the video stream.")
        return

    frame_count = 0

    print("Starting capture. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            if frame_count % SAVE_EVERY_N_FRAMES != 0:
                continue
                
            resized_frame = cv2.resize(frame, (target_width, target_height))
            
            filename = os.path.join(
                output_folder,
                "camera_img.jpg"
            )

            success = cv2.imwrite(filename, resized_frame)
            if success:
                print(f"Saved Image → {filename}")
            else:
                print(f"Error: Could not save {filename}")
            break

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")

    finally:
        cap.release()
        print("Capture released. Done.")
        return filename
