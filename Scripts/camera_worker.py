import cv2
from util import color_for_id
from global_id import cosine_similarity
import param as param

class TrackletState:
    #Per-camera EMA embedding for a tracklet.

    def __init__(self, embedding, alpha=0.9):
        self.ema_embedding = embedding.copy()
        self.alpha = alpha

    def update(self, embedding):
        self.ema_embedding = self.alpha * self.ema_embedding + (1 - self.alpha) * embedding


class CameraWorker:
    #Camera worker handling detection, tracking, tracklets, and global ID assignment.

    def __init__(
        self,
        source,
        camera_name,
        detector,
        tracker,
        global_id_store,
        frame_buffer,
        shutdown_event,
        confidence_threshold=param.confidence_threshold,
        tracklet_ema_alpha=param.tracklet_ema_alpha,
    ):
        self.source = source
        self.camera_name = camera_name
        self.detector = detector
        self.tracker = tracker
        self.global_id_store = global_id_store
        self.frame_buffer = frame_buffer
        self.shutdown_event = shutdown_event
        self.confidence_threshold = confidence_threshold
        self.tracklet_ema_alpha = tracklet_ema_alpha

        self.tracklets = {}   
        self.track_to_gid = {}   

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.camera_name}")
            return

        try:
            while not self.shutdown_event.is_set():
                success, frame = cap.read()
                if not success:
                    break

                # 1. Detection 
                detections = self.detector.detect(frame)
                detections = [d for d in detections if d[1] >= self.confidence_threshold]

                # 2. DeepSORT Tracking 
                tracks = self.tracker.update(detections, frame)
                used_gids = set()

                for track in tracks:
                    if not track.is_confirmed() or not track.features:
                        continue

                    track_id = track.track_id
                    embedding = track.features[-1]

                    # Tracklet EMA 
                    if track_id not in self.tracklets:
                        self.tracklets[track_id] = TrackletState(embedding, self.tracklet_ema_alpha)
                    else:
                        self.tracklets[track_id].update(embedding)

                    tracklet_embedding = self.tracklets[track_id].ema_embedding

                    #4. Global ID Assignment
                    if track_id not in self.track_to_gid:
                        gid, sim_score = self.global_id_store.match_tracklet(tracklet_embedding, exclude_gids=used_gids)
                        if gid is None:
                            # fallback: create new global ID
                            gid = self.global_id_store.create_new(tracklet_embedding)
                            sim_score = 1.0
                        self.track_to_gid[track_id] = gid
                    else:
                        gid = self.track_to_gid[track_id]
                        sim_score = max(cosine_similarity(tracklet_embedding, e)
                                        for e, _ in self.global_id_store._store[gid].embedding_buffer)

                    used_gids.add(gid)

                    # 5. Update Global ID with frame embedding 
                    self.global_id_store.update_frame_embedding(embedding, gid)

                    # 6. Visualization
                    x1, y1, x2, y2 = map(int, track.to_tlbr())
                    color = color_for_id(gid)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {gid} {sim_score:.2f}"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    tx, ty = x1, y2 - 5
                    cv2.rectangle(frame, (tx, ty - th - 6), (tx + tw + 4, ty), color, -1)
                    cv2.putText(frame, label, (tx + 2, ty - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # 7. Output Frame 
                self.frame_buffer.update(self.camera_name, frame)

        finally:
            cap.release()
            print(f"[INFO] Camera closed: {self.camera_name}")
