from worker.tracker import read_ongoing, update_ongoing

import requests

master_host = "192.168.0.23"
master_port = 5001
url_prefix = f"http://{master_host}:{master_port}"

current_job = read_ongoing()
print(f"Current job: {current_job}")
if current_job != "":
    response = requests.post(f"{url_prefix}/job/unassign", params={"job":current_job})
    if response.text == "OK":
        print("Job successfully unassigned")
    else:
        print(response.text)