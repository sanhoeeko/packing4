import os
from datetime import datetime

import psutil


def getProcessIds():
    process_ids = []
    for proc in psutil.process_iter(['pid']):
        process_ids.append(proc.info['pid'])
    return process_ids


class MyThreadRecord:
    def __init__(self, user_name: str, num_threads):
        self.user_name = user_name
        self.num_threads = num_threads
        self.file_path = "/home/share/thread.txt"
        self.pid_str = f"[{os.getpid()}]"

    def __enter__(self):
        if not os.path.exists(self.file_path):
            return
        # check existing records: if a process does not exist, delete the record
        current_pids = getProcessIds()
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        with open(self.file_path, 'w') as file:
            for line in lines:
                if line.startswith('['):
                    pid = int(line[1:line.index(']')])
                    if pid not in current_pids:
                        continue
                file.write(line)
        # add a record
        with open(self.file_path, 'a') as f:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{self.pid_str} {self.user_name} {self.num_threads} threads, {time_str}\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not os.path.exists(self.file_path):
            return
        # delete the line that starts with the current PID
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        with open(self.file_path, 'w') as f:
            for line in lines:
                if not line.startswith(self.pid_str):
                    f.write(line)
