# data_collector/recorder.py
import time
import cv2
import mss
import numpy as np
import threading
from pynput import mouse, keyboard
from .config import FPS, RESIZE_FACTOR, CHUNK_SIZE
from .io_utils import save_chunk

class DataRecorder:
    def __init__(self):
        self.running = False
        self.mouse_events = []
        self.keyboard_events = []
        
        # Buffer to hold data before saving
        self.frame_buffer = []
        self.mouse_buffer = []
        self.key_buffer = []
        
        # Screen capture setup
        self.sct = mss.mss()
        # Monitor 0 is the "All in One" bounding box for multi-monitor setups
        self.monitor = self.sct.monitors[0] 
        
        # Thread locks
        self.lock = threading.Lock()

    def on_move(self, x, y):
        with self.lock:
            self.mouse_events.append({'t': time.time(), 'action': 'move', 'x': x, 'y': y})

    def on_click(self, x, y, button, pressed):
        with self.lock:
            action = 'pressed' if pressed else 'released'
            self.mouse_events.append({
                't': time.time(), 
                'action': 'click', 
                'x': x, 
                'y': y, 
                'button': str(button),
                'state': action
            })

    def on_press(self, key):
        with self.lock:
            try:
                k = key.char
            except AttributeError:
                k = str(key)
            self.keyboard_events.append({'t': time.time(), 'action': 'press', 'key': k})

    def capture_cycle(self):
        """The main loop that runs at X FPS"""
        print(f"Recorder started. Resolution: {self.monitor['width']}x{self.monitor['height']} @ {FPS} FPS")
        print("Press Ctrl+C in terminal to stop.")
        
        try:
            while self.running:
                start_time = time.time()

                # 1. Grab Screen
                # mss returns raw pixels. We convert to numpy.
                sct_img = self.sct.grab(self.monitor)
                img = np.array(sct_img)
                
                # Drop Alpha channel (BGRA -> BGR) to save space
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Resize if configured
                if RESIZE_FACTOR != 1.0:
                    width = int(img.shape[1] * RESIZE_FACTOR)
                    height = int(img.shape[0] * RESIZE_FACTOR)
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                # 2. Grab Inputs since last frame
                with self.lock:
                    current_mouse = self.mouse_events[:]
                    current_key = self.keyboard_events[:]
                    self.mouse_events = []    # Reset temporary list
                    self.keyboard_events = [] # Reset temporary list

                # 3. Add to Buffer
                self.frame_buffer.append(img)
                self.mouse_buffer.append(current_mouse)
                self.key_buffer.append(current_key)

                # 4. Check if Buffer is full -> Save to Disk
                if len(self.frame_buffer) >= CHUNK_SIZE:
                    # Save in a separate thread so we don't pause the recording
                    threading.Thread(target=save_chunk, args=(
                        self.frame_buffer[:], 
                        self.mouse_buffer[:], 
                        self.key_buffer[:]
                    )).start()
                    
                    # Clear buffers
                    self.frame_buffer = []
                    self.mouse_buffer = []
                    self.key_buffer = []

                # 5. Maintain FPS
                elapsed = time.time() - start_time
                sleep_time = (1.0 / FPS) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Save whatever is left in buffers
            if self.frame_buffer:
                print("Saving remaining data...")
                save_chunk(self.frame_buffer, self.mouse_buffer, self.key_buffer)

    def start(self):
        self.running = True
        
        # Start Input Listeners (Non-blocking)
        m_listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        k_listener = keyboard.Listener(on_press=self.on_press)
        
        m_listener.start()
        k_listener.start()

        # Run Main Vision Loop
        self.capture_cycle()

        m_listener.stop()
        k_listener.stop()

if __name__ == "__main__":
    rec = DataRecorder()
    rec.start()