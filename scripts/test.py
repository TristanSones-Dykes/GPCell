import numpy as np
import time
import gpflow as gp
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


# Modified _predict function to improve performance by reducing computation time per model
def _predict(x, y):
    # Using a smaller number of iterations for the optimizer to speed up the process
    kern = gp.kernels.RBF()
    model = gp.models.GPR((x, y), kern)
    # Training model with fewer iterations for speed
    gp.optimizers.Scipy().minimize(
        model.training_loss,
        model.trainable_variables,
        options=dict(maxiter=20),  # Reduced iterations for speed
    )
    return model


# Function for processing a batch of data with multithreading
def process_batch(batch_X, batch_Y, max_workers):
    batch_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_predict, x, y) for x, y in zip(batch_X, batch_Y)]
        for future in futures:
            batch_results.append(future.result())
    return batch_results


# Function to predict using forked child processes
def _async_with_fork_processes(X, Y, max_processes=4, max_workers=8, batch_size=5):
    results = []

    # Function that each child process will execute
    def worker(batch_X, batch_Y, result_queue):
        result = process_batch(batch_X, batch_Y, max_workers)
        result_queue.put(result)

    # Using multiprocessing with fork
    result_queue = multiprocessing.Queue()
    processes = []

    for i in range(0, len(X), batch_size):
        batch_X = X[i : i + batch_size]
        batch_Y = Y[i : i + batch_size]
        p = multiprocessing.Process(
            target=worker, args=(batch_X, batch_Y, result_queue)
        )
        p.start()
        processes.append(p)

    # Collecting results from all processes
    for p in processes:
        p.join()
        results.extend(result_queue.get())

    return results


# Original iterative approach for reference
def _iterative(X, Y):
    results = []
    for x, y in zip(X, Y):
        results.append(_predict(x, y))
    return results


if __name__ == "__main__":
    X = np.random.normal(size=[50, 100, 1])
    Y = np.random.normal(size=[50, 100, 1])

    # Run and time the forked multiprocessing and multithreaded approach
    start_time = time.time()
    models_async_forked = _async_with_fork_processes(
        X, Y, max_processes=4, max_workers=8, batch_size=5
    )
    async_forked_time = time.time() - start_time

    # Run and time the iterative approach
    start_time = time.time()
    models_iterative = _iterative(X, Y)
    iterative_time = time.time() - start_time

    # Comparison summary
    print("\nComparison:")
    print(
        f"Async approach with forked processes and multithreading: {async_forked_time:.2f} seconds"
    )
    print(f"Iterative approach: {iterative_time:.2f} seconds")

    # Summary of the models
    gp.utilities.print_summary(models_async_forked[0])
    gp.utilities.print_summary(models_iterative[0])
