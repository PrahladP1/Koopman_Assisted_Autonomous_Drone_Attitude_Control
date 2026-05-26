import numpy as np
import torch
import time
from Inference.load_model import load_trained_model
from Inference.predict import KoopmanPredict
from pathlib import Path

def main():
    BASE_DIR = Path.home()/"PycharmProjects"/"MyProjects"
    MODEL_PATH = BASE_DIR/"models"/"best_model.pth"
    device = torch.device("cude" if torch.cuda.is_available() else "cpu")
    model, config = load_trained_model(MODEL_PATH, device)
    predictor = KoopmanPredict(model, config, device)

    x = np.random.randn(15)
    u = np.random.randn(4)

    for _ in range(1000):
        predictor.predict_step(x, u)

    N = 5000
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        predictor.predict_step(x, u)
        t1 = time.perf_counter()
        times.append(t1-t0)
    times = np.array(times)
    mean_ms = 1000*np.mean(times)
    std_ms = 1000*np.std(times)
    worst_ms = 1000*np.max(times)
    freq = (1.0)/(np.mean(times))
    print(f"Mean Latency: {mean_ms:.4f} ms")
    print(f"Standard Deviation: {std_ms:.4f} ms")
    print(f"Worst Case: {worst_ms:.4f} ms")
    print(f"Frequency: {freq:.4f} ms")

if __name__ == "__main__":
    main()