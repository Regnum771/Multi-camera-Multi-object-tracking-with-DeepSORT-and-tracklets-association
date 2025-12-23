from pymongo import MongoClient
import param

def bulk_insert_frames(frames, mongo_uri=param.mongo_uri,
                       db_name=param.db_name,
                       collection_name=param.collection_name):
    """
    Insert all processed frame data into MongoDB in bulk.
    
    Each frame document has:
    {
        'camera_name': str,
        'frame_number': int,
        'detections': [
            {
                'bbox': [x, y, w, h],
                'confidence': float,
                'class_id': int,
                'tracklet_id': int,
                'global_id': int,
                'embedding': list (optional)
            },
            ...
        ]
    }
    """
    if not frames:
        print("[INFO] No frames to insert.")
        return

    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        client.admin.command("ping")
        print("[INFO] MongoDB Connected successfully.")
        # other application code
        client.close()
    except Exception as e:
        raise Exception(
            "[ERR] The following error occurred: ", e)


    # Convert embeddings to lists if they are numpy arrays
    for f in frames:
        for d in f["detections"]:
            if "embedding" in d and d["embedding"] is not None:
                d["embedding"] = d["embedding"].tolist()

    collection.insert_many(frames)
    print(f"[INFO] Inserted {len(frames)} frames into MongoDB.")