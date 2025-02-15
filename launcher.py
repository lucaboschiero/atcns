import subprocess
import time
import os

# Define launcher execution log path
execution_log_path = "./logs/execution_log_FMnist_3attackers.txt"

# Define script and argument sets
script_path = "./main.py"

aggRule = ["mstold", "foolsgold", "density", "mst", "kmeans"]
device = "cpu"
attacks = "backdoor/labelflipping/multilabelflipping"
epochs = 30
total_clients = 40
attacker_percentage = [10, 20, 30, 40, 50, 60, 70]
#attacker_percentage = [50, 60, 70]
labelflipping_percentage = [33]
dataset = "fmnist"

# Open a file for logging
with open(execution_log_path, "a") as log_file:
    # Log header
    log_file.write("Execution Log:\n")
    log_file.write("=========================================\n")

for percentage in attacker_percentage:
    num_attacker = int((total_clients * percentage) / 100)
    for lfpercentage in labelflipping_percentage:
        num_labelflipping_attacker = int((num_attacker * lfpercentage) / 100 + 0.5) 
        # Adding 0.5 ensures proper rounding to the nearest integer:  
        # - If the result is 7.5 or higher, it rounds up to 8.  
        # - If the result is 7.4 or lower, it rounds down to 7.
        num_multiLabelflipping_attacker = int((num_attacker * lfpercentage) / 100 + 0.5)
        num_backdoor_attacker = num_attacker - num_labelflipping_attacker - num_multiLabelflipping_attacker

        #print("Attacker: ", num_attacker)
        #print("Single LF: ", num_labelflipping_attacker)
        #print("Multi LF: ", num_multiLabelflipping_attacker)
        #print("Backdoor: ", num_backdoor_attacker)
        for ar in aggRule:
            # Construct arguments
            args = [
                "python", script_path,
                "--AR", str(ar),
                "--device", device,
                "--attacks", attacks,
                "--save_model_weights",
                "--n_attacker_labelFlipping", str(num_labelflipping_attacker),
                "--n_attacker_multilabelFlipping", str(num_multiLabelflipping_attacker),
                "--n_attacker_backdoor", str(num_backdoor_attacker),
                "-n", str(total_clients),
                "--epochs", str(epochs),
                "--dataset", dataset
            ]

            # Measure execution time
            start_time = time.time()
            result = subprocess.run(args)

            try:
                with open("./logs/detection_time.txt", "r") as f:
                    detection_time = f.read().strip()
                print(f"Execution Time: {detection_time} seconds")

                # Delete the file after reading
                os.remove("./logs/detection_time.txt")
                print("File deleted successfully.")
            except FileNotFoundError:
                print(f"Error: file ./logs/detection_time.txt not found!")
            
            with open(execution_log_path, "a") as log_file:
                # Log the arguments
                log_file.write(f"Running with parameters:\n")
                log_file.write(f"aggRule: {ar}, attacker_percentage: {percentage}, labelflipping_percentage: {lfpercentage}, num_labelflipping_attacker: {num_labelflipping_attacker}, num_backdoor_attacker: {num_backdoor_attacker}\n")
                
                end_time = time.time()
                execution_time = (end_time - start_time) / 60
                
                # Log the result and execution time
                if result.returncode != 0:
                    log_file.write(f"Result: Failed\n")
                else:
                    log_file.write(f"Result: Successful\n")
                
                log_file.write(f"Execution Time: {execution_time:.2f} minutes   ----   Detection Time: {detection_time} seconds\n")
                log_file.write("-----------------------------------------\n")
