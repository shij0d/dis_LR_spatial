import multiprocessing
import time
import os

# A simple CPU-bound task for demonstration
def cpu_task(x):
    print(f"Process ID: {os.getpid()} is working on {x}")
    result = 0
    for i in range(10**7):
        result += i * x
    return result

if __name__ == '__main__':
    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    # Create a pool using 75% of the available cores
    pool_size = num_cores // 4 * 3
    print(f"Using {pool_size} cores")

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=pool_size) as pool:
        # Apply CPU-bound tasks in parallel
        start_time = time.time()
        results = pool.map(cpu_task, range(10))

        # Wait for all processes to complete
        pool.close()
        pool.join()

    # Display results and time taken
    print(f"Results: {results}")
    print(f"Time taken: {time.time() - start_time} seconds")
