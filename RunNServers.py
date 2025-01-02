import subprocess
import sys

processes = []

for index in range(int(sys.argv[1])):
    process = subprocess.Popen(["python", "server.py", str(5000 + index)])
    processes.append(process)

# Wait for all processes to finish
for process in processes:
    process.wait()
