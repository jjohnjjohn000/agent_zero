# data_collector/io_utils.py
import os
import time
import numpy as np
from datetime import datetime
from .config import DATA_DIR, COMPRESSION_LEVEL

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def save_chunk(frames, mouse_data, keyboard_data):
    """
    Saves a chunk of recording data to disk.
    frames: List of numpy arrays (images)
    mouse_data: List of dicts
    keyboard_data: List of dicts
    """
    ensure_data_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.npz"
    filepath = os.path.join(DATA_DIR, filename)

    # Convert lists to numpy arrays for storage
    # We stack images into a 4D array: (Time, Height, Width, Channels)
    frames_np = np.stack(frames)
    
    # We prefer saving inputs as generic objects to preserve structure
    # pickle=True is required for non-numeric data arrays
    np.savez_compressed(
        filepath,
        images=frames_np,
        mouse=np.array(mouse_data, dtype=object),
        keyboard=np.array(keyboard_data, dtype=object)
    )
    
    print(f"[Storage] Saved chunk: {filename} | {len(frames)} frames")