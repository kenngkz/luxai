from base_utils import path_join
from constants import MASTER_CACHE_DIR, JOBS_QUEUE_FILE

from collections import deque
import os

class JobManager:
    def __init__(self, queue_file=path_join(MASTER_CACHE_DIR, JOBS_QUEUE_FILE)):
        self.queue_file = queue_file
        self.read_queue()

    # Queue
    
    def clear_queue(self):
        with open(self.queue_file, 'w') as f:
            f.write('')
        self.queue = deque([])

    def get_top_job(self):
        return self.queue

    def read_queue(self):
        if os.path.exists(self.queue_file):
            with open(self.queue_file, 'r') as f:
                file = f.read()
                if file == "":
                    self.queue = deque([])
                else:
                    self.queue = deque([eval(job) for job in file.split('\n')])
        else:
            self.clear_queue()
        return self.queue

    def _update_queue(self):
        with open(self.queue_file, 'w') as f:
            f.write('\n'.join([str(job) for job in self.queue]))

    def add_queue(self, job_type, job_args, job_info=None):
        job = {'type':job_type, 'args':job_args, 'info':job_info}
        if job_type == 'train':
            self.queue.append(job)
        else:
            self.queue.appendleft(job)
        self._update_queue()
        return job
        
    # User functions

    def assign_job(self):
        '''
        Returns the top job, removes it from queue, update queue file, adds it to ongoing, updates ongoing file.
        '''
        if len(self.queue) > 0:
            job = self.queue.popleft()
            self._update_queue()
            return job
        else:
            return None