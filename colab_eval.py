'''
Starts a FileTransferServer to enable recieving files from 
'''
import os
import shutil
import time

from file_transfer.server import FTServer
from file_transfer.client import Client
from stage import eval_stage
from utility import get_benchmarks

# Change this for authentication reasons
SERVER_PASSWORD = '159fg945vk9'

RECEIVE_FOLDER = "../modelcp/cache/"
POOL_FOLDER = "../modelcp/pool/"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = "5001"

OTHER_PASSWORD = 'bia48ga9b48mn'
OTHER_HOST = ""
OTHER_PORT = ""


n_select = 10

def start_server(password, transfer_folder=RECEIVE_FOLDER, server_host=SERVER_HOST, server_port=SERVER_PORT):
    server = FTServer(password, transfer_folder, server_host, server_port)
    server.run()
    return server

def stop_server(password, server_host=SERVER_HOST, server_port=SERVER_PORT):
    client = Client(password, server_host, server_port)
    client.connect()
    client.close_server()

def start_client(password, server_host=OTHER_HOST, server_port=OTHER_PORT):
    pass

def main():
    server = start_server(SERVER_PASSWORD, RECEIVE_FOLDER, SERVER_HOST, SERVER_PORT)
    # implement a loop to check for new models downloaded into cache. 
    # If there are new models, evaluate the model and copy files to pool stage (get stage from fileinfo.json)
    while True:
        if not server.open:  # if server is closed: CLOSE_SERVER request sent from client, quit
            break
        cache_files = os.listdir(RECEIVE_FOLDER)
        if len(cache_files) != 0:
            stage_paths = set()
            for child in cache_files:
                with open(RECEIVE_FOLDER + 'fileinfo.json', 'r') as f:
                    fileinfo = eval(f.read())
                stage = fileinfo[child]  # get the stage name
                prev_stage = stage[:-1] + str(int(stage[-1]) - 1)
                stage_path = POOL_FOLDER + stage + '/'  # define stage_path
                
                stage_paths.add(stage_path)
                if not os.path.exists(stage_path):
                    os.makedirs(stage_path)
                shutil.move(RECEIVE_FOLDER + child, stage_path + child)
            for stage_path in stage_paths:
                prev_stage_path = POOL_FOLDER + prev_stage + '/'
                benchmark_models = get_benchmarks(prev_stage)
                assert len(benchmark_models) > 0, f"FileNotFound: benchmark_models.txt not found in {prev_stage}"
                eval_stage(stage_path, n_select, benchmark_models, suppress_prints=True)  # suppress prints = True does not print skipping model eval message                    
            # TODO: send seed models to the other device
        else:
            time.sleep(1)