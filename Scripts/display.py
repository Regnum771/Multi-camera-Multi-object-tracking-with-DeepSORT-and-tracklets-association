import cv2
import time

def display_loop(frame_buffer, camera_names, shutdown_event, output_path=None, fps=30):
    writer = None
    frame_interval = 1.0 / fps
    last_write_time = time.time()

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

            if output_path and writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            cv2.imshow("Multi-Camera Tracking", combined)

            now = time.time()
            if writer and (now - last_write_time) >= frame_interval:
                writer.write(combined)
                last_write_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                shutdown_event.set()
                break

        time.sleep(0.1)
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        shutdown_event.set()
        print("[INFO] Display loop terminated.")
