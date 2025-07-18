import multiprocessing as mp
import numpy as np
import os
import sys
import subprocess
from multiprocessing.connection import Connection as Conn


class Worker(mp.Process):
    def __init__(self, pipe: Conn, device_id: int):
        super(Worker, self).__init__()

        self.pipe = pipe

        self.device_id = device_id

    def run(self):

        while True:

            command, args = self._recv_message()

            if command == "run_cmd":
                task_id, cmd = args

                print(f"> [task {task_id}] Running...")
                
                my_env = os.environ.copy()
                my_env["CUDA_VISIBLE_DEVICES"] = f"{self.device_id}"

                print(cmd.split(" "))
                subprocess.run(cmd.split(" "), env = my_env) 

                print(f"> [task {task_id}] Done!")

                self._send_message("run_cmd", None)

            elif command == "kill":
                return

            else:
                raise NotImplementedError()

    def _send_message(self, command, kwargs):
        self.pipe.send((command, kwargs))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, kwargs = self.pipe.recv()

        return command, kwargs


class Launcher(mp.Process):
    def __init__(self, device_ids):
        super(Launcher, self).__init__()

        self.device_ids = device_ids

        self.num_workers = len(device_ids)

        self.workers = []
        self.pipes = []
        for worker_idx in range(self.num_workers):
            parent_pipe, child_pipe = mp.Pipe()
            worker = Worker(child_pipe, self.device_ids[worker_idx])

            self.workers.append(worker)
            self.pipes.append(parent_pipe)

        for worker in self.workers:
            worker.start()

    def run_tasks(self, tasks):

        workers_status = np.zeros([self.num_workers], dtype = bool)

        global_task_id = 0
        rets = [None for _ in range(len(tasks))]

        while np.any(workers_status) or len(tasks) > 0:

            # Check for completed tasks
            for worker_idx in range(self.num_workers):
                command, ret_kwargs = self._recv_message_nonblocking(worker_idx)
                if command is not None:
                    workers_status[worker_idx] = False
                else:
                    continue

                if command == "run_cmd":
                    pass
                else:
                    raise NotImplementedError()

            # Assign tasks to empty workers
            while len(tasks) > 0 and not np.all(workers_status):
                worker_idx = np.where(~workers_status)[0][0]

                command, task_args = tasks[global_task_id]
                task_id = global_task_id
                global_task_id += 1

                self._send_message(worker_idx, command, (task_id, *task_args))

                workers_status[worker_idx] = True

        return rets

    def _send_message(self, worker_idx, command, args = None):
        pipe = self.pipes[worker_idx]

        pipe.send((command, args))

    def _recv_message_nonblocking(self, worker_idx):
        pipe = self.pipes[worker_idx]

        if not pipe.poll():
            return None, None

        command, args = pipe.recv()

        return command, args

    def _recv_message(self, worker_idx):
        pipe = self.pipes[worker_idx]

        pipe.poll(None)

        command, args = pipe.recv()

        return command, args

    def __del__(self):
        for worker_idx in range(self.num_workers):
            self._send_message(worker_idx, "kill")