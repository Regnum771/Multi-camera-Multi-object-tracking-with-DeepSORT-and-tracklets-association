from deep_sort_realtime.deepsort_tracker import DeepSort


class ReIDTracker:
    #DeepSORT tracker with appearance embeddings.
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            embedder="mobilenet",
            half=True,
            bgr=True
        )

    def update(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)