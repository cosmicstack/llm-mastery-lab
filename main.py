from tqdm import tqdm
import numpy as np
import time

def check_tqdm():
    count = 0
    for i in tqdm(range(100), desc="Adding random normal variables"):
        time.sleep(0.2)
        count += np.random.normal()
    return count

if __name__ == "__main__":
    result = check_tqdm()
    print(f"Result: {result}")
    print("TQDM is working correctly.")