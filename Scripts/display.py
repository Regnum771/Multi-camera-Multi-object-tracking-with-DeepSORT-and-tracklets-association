import cv2
import time

def display_loop(frame_buffer, camera_names, shutdown_event, fps=30):
    try:
        while not shutdown_event.is_set():
            frames = []
            missing = False
            for name in camera_names:
                frame = frame_buffer.get(name)
                if frame is None:
                    missing = True
                    break
                frames.append(frame)

            if missing:
                time.sleep(0.01)
                continue

            combined = cv2.hconcat(frames)

            cv2.imshow("Multi-Camera Tracking", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                shutdown_event.set()
                break

        time.sleep(0.1)
    finally:
        cv2.destroyAllWindows()
        shutdown_event.set()
        print("[INFO] Display loop terminated.")
