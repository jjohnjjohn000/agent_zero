# data_collector/config.py
import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- RECORDING SETTINGS ---
# How many times per second to capture the screen?
# 2-5 FPS is usually enough for OS tasks. 60 FPS is overkill and eats disk space.
FPS = 4.0 

# Resize Factor: 1.0 = Native Resolution. 0.5 = Half size.
# With 3 monitors, 0.5 is a good balance between readability and disk usage.
RESIZE_FACTOR = 0.5 

# Chunk Size: How many frames to keep in RAM before writing to disk.
# 100 frames @ 4 FPS = writes every 25 seconds.
CHUNK_SIZE = 100

# --- COMPRESSION ---
# Level 1 (fastest) to 9 (smallest). 
# We want fast so we don't lag the PC.
COMPRESSION_LEVEL = 3 