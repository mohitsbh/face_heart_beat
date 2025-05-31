import numpy as np
from collections import deque
import time
from scipy.signal import butter, filtfilt

class HeartRateEstimator:
    def __init__(self, buffer_size=150, fps=30):
        self.buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.fps = fps
        self.bpm_history = []

    def reset(self):
        self.buffer.clear()
        self.timestamps.clear()
        self.bpm_history.clear()

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], btype='band')

    def bandpass_filter(self, data, lowcut=0.75, highcut=3.0, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, self.fps, order)
        return filtfilt(b, a, data)

    def update(self, green_intensity):
        self.buffer.append(green_intensity)
        self.timestamps.append(time.time())

    def get_heart_rate(self):
        if len(self.buffer) < self.buffer.maxlen:
            return None

        signal = np.array(self.buffer)
        signal -= np.mean(signal)

        filtered = self.bandpass_filter(signal)

        fft = np.fft.rfft(filtered)
        freqs = np.fft.rfftfreq(len(filtered), d=1 / self.fps)

        peak_idx = np.argmax(np.abs(fft))
        peak_freq = freqs[peak_idx]
        bpm = peak_freq * 60

        if 45 <= bpm <= 180:
            self.bpm_history.append(bpm)
            if len(self.bpm_history) > 5:
                self.bpm_history = self.bpm_history[-5:]
            return round(np.median(self.bpm_history), 2)
        return None
