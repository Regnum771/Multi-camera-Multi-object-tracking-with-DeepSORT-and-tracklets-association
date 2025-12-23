import threading
import cv2

from detector import YOLODetector
from camera_worker import CameraWorker
from global_id import GlobalIDStore
from frame_buffer import FrameBuffer
from tracker import ReIDTracker
from display import display_loop
from data_exporter import bulk_insert_frames

def main():
    shutdown_event = threading.Event()

    detector = YOLODetector("yolov8n.pt")
    global_id_store = GlobalIDStore()
    frame_buffer = FrameBuffer()

    # Replace 0 and 1 with actual video sources if there's no webcam
    cameras = [
        (0, "Camera 1"), 
        (1, "Camera 2"),
    ]

    camera_workers = []
    threads = []

    # Start camera workers
    for source, name in cameras:
        worker = CameraWorker(
            source=source,
            camera_name=name,
            detector=detector,
            tracker=ReIDTracker(),
            global_id_store=global_id_store,
            frame_buffer=frame_buffer,
            shutdown_event=shutdown_event,
        )
        camera_workers.append(worker)

        t = threading.Thread(target=worker.run)
        t.start()
        threads.append(t)

    try:
        display_loop(
            frame_buffer,
            [name for _, name in cameras],
            shutdown_event=shutdown_event,
            fps=60
        )
    finally:
        # Shutdown all threads
        shutdown_event.set()
        for t in threads:
            t.join()
        cv2.destroyAllWindows()
        print("[INFO] Program terminated cleanly.")

        # Export detection data to mongoDB
        detection_data = []
        for worker in camera_workers:
            detection_data.extend(worker.processed_frames)

        if detection_data:
            bulk_insert_frames(detection_data)
            print(f"[INFO] Exported {len(detection_data)} frames to MongoDB.")


if __name__ == "__main__":
    main()

