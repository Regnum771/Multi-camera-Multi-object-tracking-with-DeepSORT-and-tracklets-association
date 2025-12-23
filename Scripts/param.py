""" GlobalID parameters """
# Minimum embedding similarity to consider a match
similarity_threshold = 0.75
# Minimum superiority over other candidates to assign ID
relative_margin = 0.05
# Maximum lifetime of a global ID before eviction .. change depending on fps
max_age_seconds = 60.0  
# Time after last seen to consider a tracklet inactive.. change depending on fps
inactive_age_seconds = 30.0        
# Weight for tracklet-level EMA when updating global embedding
tracklet_weight = 0.9              
# Weight for single-frame embedding in EMA update
frame_weight = 0.3                 

""" CameraWorker parameters """
# Minimum detection confidence to keep a bounding box
confidence_threshold = 0.6         
# EMA smoothing factor for per-tracklet embeddings
tracklet_ema_alpha = 0.9

"""data exporter parameters"""
# insert your mongodb connection string here
mongo_uri = ""
db_name = "MCTMT"
collection_name = "frames"