import subprocess
import time

# Define launcher execution log path
execution_log_path = "./logs/execution_log.txt"

# Define script and argument sets
script_path = "./main.py"

aggRule = ["mst", "kmeans", "mstold", "foolsgold", "density"]
#aggRule = ["kmeans"]
device = "cpu"
attacks = "backdoor/labelflipping"
epochs = 30
total_clients = 40
attacker_percentage = [10, 20, 30, 40, 50, 60, 70]
labelflipping_percentage = [25, 50, 75]
dataset = "cifar"

# Open a file for logging
with open(execution_log_path, "a") as log_file:
    # Log header
    log_file.write("Execution Log:\n")
    log_file.write("=========================================\n")

for percentage in attacker_percentage:
    num_attacker = int((total_clients * percentage) / 100)
    for lfpercentage in labelflipping_percentage:
        num_labelflipping_attacker = int((num_attacker * lfpercentage) / 100)
        num_backdoor_attacker = num_attacker - num_labelflipping_attacker
        for ar in aggRule:
            # Construct arguments
            args = [
                "python3", script_path,
                "--AR", str(ar),
                "--device", device,
                "--attacks", attacks,
                "--save_model_weights",
                "--n_attacker_labelFlipping", str(num_labelflipping_attacker),
                "--n_attacker_backdoor", str(num_backdoor_attacker),
                "-n", str(total_clients),
                "--epochs", str(epochs),
                "--dataset", dataset,
                "--inner_epochs", "5"
            ]
            
            with open(execution_log_path, "a") as log_file:
                # Log the arguments
                log_file.write(f"Running with parameters:\n")
                log_file.write(f"aggRule: {ar}, attacker_percentage: {percentage}, labelflipping_percentage: {lfpercentage}, num_labelflipping_attacker: {num_labelflipping_attacker}, num_backdoor_attacker: {num_backdoor_attacker}\n")
                
                # Measure execution time
                start_time = time.time()
                result = subprocess.run(args)
                end_time = time.time()
                execution_time = (end_time - start_time) / 60
                
                # Log the result and execution time
                if result.returncode != 0:
                    log_file.write(f"Result: Failed\n")
                else:
                    log_file.write(f"Result: Successful\n")
                
                log_file.write(f"Execution Time: {execution_time:.2f} minutes\n")
                log_file.write("-----------------------------------------\n")
