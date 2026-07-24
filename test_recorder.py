import os
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
SAMPLE_RATE = 44100      # Standard CD quality audio
BLOCK_SIZE = 1024        # Number of audio frames per buffer chunk
SILENCE_DURATION = 5.0   # Seconds of silence required to split
SILENCE_THRESHOLD = 0.02 # Volume threshold (0.0 to 1.0). Raise this if your room is noisy.

# --- INITIALIZE WHISPER MODEL ---
print("Loading Whisper Speech-to-Text Model...")
# The "base" model strikes a fantastic balance between speed and precision for English, Russian, and Armenian.
# If your Mac has an M-series chip, it will run incredibly well on CPU using int8 quantization.
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model loaded successfully.")

# Thread-safe queue to pass saved files from the audio loop to the transcription pipeline
transcription_queue = queue.Queue()

# State variables
recording_frames = []
is_recording = False
silence_start_time = None
file_counter = 1

def audio_callback(indata, frames, time_info, status):
    """
    This function is called automatically by sounddevice for every chunk of audio.
    indata: Data coming directly from your microphone
    """
    global recording_frames, is_recording, silence_start_time, file_counter
    
    if status:
        print(status)
    
    # Calculate Volume Level (Root Mean Square of the input signal float values)
    volume_norm = np.sqrt(np.mean(indata**2))
    
    if volume_norm > SILENCE_THRESHOLD:
        # Sound detected!
        if not is_recording:
            print(f"\n[Sound Detected] Active: Recording File #{file_counter}...")
            is_recording = True
        
        recording_frames.append(indata.copy())
        silence_start_time = None  # Reset the silence clock
    else:
        # Silence detected!
        if is_recording:
            recording_frames.append(indata.copy())  # Keep tracking the silent frames
            
            if silence_start_time is None:
                silence_start_time = time.time()
                
            elapsed_silence = time.time() - silence_start_time
            print(f"Silence detected... {elapsed_silence:.1f}s / {SILENCE_DURATION}s", end="\r")
            
            if elapsed_silence >= SILENCE_DURATION:
                print(f"\n[Silence Limit Reached] Splitting audio...")
                
                # Convert list of blocks into one continuous numpy audio array
                audio_array = np.concatenate(recording_frames, axis=0)
                
                # Clean up: Trim the trailing 5 seconds of silence from the file
                trim_samples = int(SAMPLE_RATE * SILENCE_DURATION)
                if len(audio_array) > trim_samples:
                    audio_array = audio_array[:-trim_samples]
                
                # Save the file to the current folder
                filename = f"audio_clip_{file_counter}.wav"
                sf.write(filename, audio_array, SAMPLE_RATE)
                print(f"Saved successfully: {os.path.abspath(filename)}")
                
                # Send the filename over to the processing queue
                transcription_queue.put(filename)
                
                # Reset tracking values for the next phrase
                file_counter += 1
                recording_frames = []
                is_recording = False
                silence_start_time = None

def transcription_worker():
    """ Runs constantly in the background to handle speech-to-text processing """
    while True:
        # Pull the next file out of the queue (blocks thread until an item is available)
        audio_filename = transcription_queue.get()
        if audio_filename is None: 
            break
            
        print(f"\n[Transcribing] Processing {audio_filename}...")
        
        try:
            # Transcribe the audio file. Whisper handles language auto-detection automatically.
            segments, info = model.transcribe(audio_filename, beam_size=5)
            
            # Combine individual text segments into a single cohesive string
            full_text = " ".join([segment.text for segment in segments]).strip()
            
            print(f" -> Detected Language: {info.language} (Confidence: {info.language_probability:.2f})")
            print(f" -> Text Result: \"{full_text}\"")
            
            # Create matching text filename (e.g., audio_clip_1.txt)
            txt_filename = os.path.splitext(audio_filename)[0] + ".txt"
            
            # Save the text data to the directory
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(full_text)
                
            print(f"Saved transcript to: {os.path.abspath(txt_filename)}")
            
        except Exception as e:
            print(f"Error transcribing {audio_filename}: {e}")
            
        # Inform the queue that the item processing is complete
        transcription_queue.task_done()

# Spin up the asynchronous transcription background thread
worker_thread = threading.Thread(target=transcription_worker, daemon=True)
worker_thread.start()

# Open the recording input stream
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    callback=audio_callback
)

with stream:
    print("\n>>> SYSTEM LIVE. Start speaking (English, Russian, or Armenian)... <<<")
    print("Press Ctrl+C in this terminal window to stop the program at any time.\n")
    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive while workers process tasks
    except KeyboardInterrupt:
        print("\n\nShutting down stream...")
        # Gracefully stop the transcription pipeline
        transcription_queue.put(None)
        worker_thread.join()
        print("Session closed. Goodbye!")