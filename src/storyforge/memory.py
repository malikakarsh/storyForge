import threading

import torch


class MPSMemoryMonitor:
    def __init__(self, interval_seconds):
        if interval_seconds <= 0:
            raise ValueError("Sampling interval must be positive.")

        self.interval = interval_seconds

        self.max_tensor_bytes = 0
        self.max_driver_bytes = 0
        self.sample_count = 0

        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def sample(self):
        tensor_bytes = torch.mps.current_allocated_memory()
        driver_bytes = torch.mps.driver_allocated_memory()

        self.max_tensor_bytes = max(
            self.max_tensor_bytes,
            tensor_bytes,
        )

        self.max_driver_bytes = max(
            self.max_driver_bytes,
            driver_bytes,
        )

        self.sample_count += 1

    def watch(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.sample()
        except Exception as error:
            self.error = error

    def start(self):
        if not torch.backends.mps.is_available():
            raise RuntimeError("This monitor requires MPS.")

        self.sample()

        self.thread = threading.Thread(
            target=self.watch,
            daemon=True,
        )

        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

        if self.error is not None:
            raise RuntimeError("Memory sampling failed.") from self.error

        self.sample()

        return {
            "max_observed_tensor_gib": (
                self.max_tensor_bytes / 2**30
            ),
            "max_observed_driver_gib": (
                self.max_driver_bytes / 2**30
            ),
            "sample_count": self.sample_count,
            "requested_sample_interval_seconds": self.interval,
        }
