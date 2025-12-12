import time
import numpy as np
from joblib import Parallel, delayed


LONG_SET_LENGTH =5_000_000
PATTERN_LENGTH = 625
NUM_REPETITIONS = 5 
NUM_CORES =8

def generate_data():
    long_set = np.random.rand(LONG_SET_LENGTH).astype(np.float32)#random long_set of numbers generated
    pattern = np.random.rand(PATTERN_LENGTH).astype(np.float32)#random pattern generated
   
    # This ensures a perfect match (error 0.0) exists
    hidden_pos =4_999_375
    long_set[hidden_pos : hidden_pos + PATTERN_LENGTH] = pattern#we introduce the pattern to ensure we find it(error==0.0)
    
    print(f"Pattern HIDDEN at   : {hidden_pos}\n")
    return long_set, pattern, hidden_pos


def calculate_sad_window(index, long_series, pattern):
    
    # Slice (view) without copying extra memory thanks to Numpy
    window = long_series[index : index + len(pattern)]
   
    return np.sum(np.abs(window - pattern))


def search_sequential(long_set, pattern):
    best_error = float('inf')#we initially set best error to infinite
    best_pos = -1#initialization to -1
    limit = len(long_set) - len(pattern) + 1
    
    for i in range(limit):
        error = calculate_sad_window(i, long_set, pattern)
        if error < best_error:
            best_error = error
            best_pos = i
            if best_error == 0.0: break
    return best_pos, best_error


def search_parallel(long_set, pattern, n_jobs):
    n_workers = n_jobs
    limit = len(long_set) - len(pattern) + 1
    
    # Data Decomposition: Split indices based on the number of workers
    total_indices = range(limit)
    chunk_size = len(total_indices) // n_workers
    chunks = [total_indices[i*chunk_size : (i+1)*chunk_size] for i in range(n_workers)]#we make a list of the divisions of work for each worker(core)
    # Ensure the last chunk covers the remainder
    chunks[-1] = total_indices[(n_workers-1)*chunk_size:]#in case some work isnt assigned to any worker

    # Worker function: executes isolated in each process
    def worker(indices):
        local_min = float('inf')
        local_pos = -1
        # Reads 'long_set' from shared memory via memmap (Zero-copy)
        for i in indices:
            err = calculate_sad_window(i, long_set, pattern)
            if err < local_min:
                local_min = err
                local_pos = i
                if local_min == 0.0: break
        return local_pos, local_min

    # Joblib configuration 
    # - n_jobs: Uses the number of cores defined in configuration.
    #- max_nbytes=1M: automatically activates memmap for large arrays.
    #- mmap_mode='r': Read-only mode to prevent overhead.
    #- batch_size='auto': Optimizes task scheduling.
    with Parallel(n_jobs=n_jobs, max_nbytes='1M', mmap_mode='r', batch_size=1) as parallel:
        results = parallel(delayed(worker)(chunk) for chunk in chunks)
    
   #Find the best result (min error) from the list of results
    return min(results, key=lambda x: x[1])##we use a lambda function to take the error and compare it


if __name__ == '__main__':#Protect the creation of the processes in the main
    # Prepare Data
    long_set, pattern, real_pos = generate_data()
        
    # Run a quick parallel search to see what it finds
    found_pos, found_error = search_parallel(long_set, pattern, n_jobs=NUM_CORES)
    
    print(f"Position Found: {found_pos}")
    print(f"Error Calculated: {found_error:.4f}")
    
    # Visual check: Print first 5 numbers to compare
    print("Visual Check (First 5 numbers):")
    sample_pattern = pattern[:5]
    sample_found = long_set[found_pos : found_pos + 5]
    
    print(f"Target Pattern:   {sample_pattern}")
    print(f"Found Sequence:   {sample_found}")


    # Time accumulators
    sum_time_seq = 0
    sum_time_par = 0
    
    print(f"Testing started ({NUM_REPETITIONS} iterations) using {NUM_CORES} cores...")
    
    for i in range(NUM_REPETITIONS):
        print(f"Processing iteration {i+1} of {NUM_REPETITIONS}...")
        
        # --- Measure Sequential ---
        start = time.time()
        search_sequential(long_set, pattern)
        sum_time_seq += (time.time() - start)
        
        # --- Measure Parallel ---
        start = time.time()
        search_parallel(long_set, pattern, n_jobs=NUM_CORES)
        sum_time_par += (time.time() - start)

    # Calculate Averages
    avg_seq = sum_time_seq / NUM_REPETITIONS
    avg_par = sum_time_par / NUM_REPETITIONS
    

    print("FINAL RESULTS (Arithmetic Mean)")
    
    print(f"Average Sequential Time: {avg_seq:.4f} s")
    print(f"Average Parallel Time ({NUM_CORES} cores):   {avg_par:.4f} s")
    