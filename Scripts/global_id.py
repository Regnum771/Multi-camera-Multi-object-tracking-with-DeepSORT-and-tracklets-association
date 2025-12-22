import time
import threading
import numpy as np
import param as param

class GlobalID:
    #Global ID with embedding buffer for cross-camera matching.

    def __init__(self, gid: int, embedding: np.ndarray):
        self.id = gid
        self.embedding_buffer = [(embedding.copy(), time.time())] 
        self.last_seen = time.time()
        self.inactive = False

    def update_embedding(self, embedding: np.ndarray, weight: float = 1.0):
        #Update embedding buffer using weighted EMA
        if self.embedding_buffer:
            prev_emb, _ = self.embedding_buffer[-1]
            new_emb = weight * prev_emb + (1 - weight) * embedding
        else:
            new_emb = embedding.copy()
        self.embedding_buffer.append((new_emb, time.time()))
        self.last_seen = time.time()
        self.inactive = False

    def prune_old(self, max_age_seconds: float):
        now = time.time()
        self.embedding_buffer = [
            (e, t) for e, t in self.embedding_buffer if now - t <= max_age_seconds
        ]
        if not self.embedding_buffer:
            self.inactive = True


class GlobalIDStore:
    #Global ID store for cross-camera ReID using tracklet + frame embeddings

    def __init__(
        self,
        similarity_threshold: float = param.similarity_threshold,
        relative_margin: float = param.relative_margin,
        max_age_seconds: float = param.max_age_seconds,
        inactive_age_seconds: float = param.inactive_age_seconds,
        tracklet_weight: float = param.tracklet_weight,
        frame_weight: float = param.frame_weight,
    ):
        self.similarity_threshold = similarity_threshold
        self.relative_margin = relative_margin
        self.max_age_seconds = max_age_seconds
        self.inactive_age_seconds = inactive_age_seconds
        self.tracklet_weight = tracklet_weight
        self.frame_weight = frame_weight

        self._next_id = 1
        self._store: dict[int, GlobalID] = {}
        self._lock = threading.Lock()

    def match_tracklet(self, embedding: np.ndarray, exclude_gids: set[int] = set()) -> tuple[int | None, float]:
        #Match embedding to active or recently inactive global IDs.
        now = time.time()
        candidates = []

        with self._lock:
            # prune old embeddings
            for gid, gid_obj in self._store.items():
                gid_obj.prune_old(self.max_age_seconds)

            # collect candidates
            for gid, gid_obj in self._store.items():
                if gid in exclude_gids:
                    continue
                if gid_obj.inactive and (now - gid_obj.last_seen) > self.inactive_age_seconds:
                    continue
                max_sim = max(cosine_similarity(embedding, e) for e, _ in gid_obj.embedding_buffer)
                candidates.append((gid, max_sim))

            if not candidates:
                return None, 0.0

            # sort by similarity
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_gid, best_sim = candidates[0]

            # relative superiority
            if len(candidates) > 1 and (best_sim - candidates[1][1]) < self.relative_margin:
                return None, best_sim

            if best_sim >= self.similarity_threshold:
                self._store[best_gid].update_embedding(embedding, weight=self.tracklet_weight)
                return best_gid, best_sim

        return None, best_sim

    def update_frame_embedding(self, embedding: np.ndarray, gid: int):
        #update global ID buffer with per-frame embedding for stronger cross-camera ReID.
        with self._lock:
            if gid in self._store:
                self._store[gid].update_embedding(embedding, weight=self.frame_weight)

    def create_new(self, embedding: np.ndarray) -> int:
        with self._lock:
            gid = self._next_id
            self._next_id += 1
            self._store[gid] = GlobalID(gid, embedding)
            return gid

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))
