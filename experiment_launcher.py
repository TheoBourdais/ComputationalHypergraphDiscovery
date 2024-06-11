import pandas as pd
import random
import subprocess
from queue import Queue, Empty
from threading import Thread
import time


def worker(device_id, queue):
    while True:
        try:
            experiment = queue.get_nowait()
        except Empty:
            break

        try:
            # Construct the command to run the script
            command = (
                f"python3 experiment.py --device={device_id} --run_index={experiment}"
            )
            # Start the experiment
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            print(f"Error running experiment {experiment} on device {device_id} : {e}")
            print("Retrying")
            try:
                # Start the experiment
                subprocess.run(command, shell=True, check=True)
            except Exception as e:
                print(
                    f"Second error running experiment {experiment} on device {device_id} : {e}"
                )
                print("Skipping")

        finally:
            queue.task_done()


def main():
    """to_analyze = list(pd.read_csv("./BCR_uniform.csv").columns[1122:])
    random.seed(42)
    random.shuffle(to_analyze)

    batch_size = 1
    to_analyze_batches = [
        list(to_analyze[i : i + batch_size])
        for i in range(0, len(to_analyze), batch_size)
    ]
    to_analyze_batches[-1] = to_analyze_batches[-1][
        : len(to_analyze_batches) % batch_size
    ]
    df_batches = pd.DataFrame(to_analyze_batches)
    df_batches.columns = [f"target {i}" for i in range(1, batch_size + 1)]
    df_batches.to_csv("batches.csv", index=False)"""
    df_batches = pd.read_csv("batches.csv")
    num_devices = 3
    max_scripts_per_device = 5
    num_experiments = len(df_batches)
    queue = Queue()

    # Fill the queues with experiments
    for i in range(num_experiments):
        queue.put(i)

    threads = []
    for device_id in range(1, num_devices + 1):
        # Create up to max_scripts_per_device threads per device
        for _ in range(max_scripts_per_device - int(device_id == 0)):
            t = Thread(target=worker, args=(device_id, queue))
            t.start()
            threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
