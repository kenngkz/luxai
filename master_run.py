"""
Run this script to call on other scripts...
This is to standardise the imports for all scripts: add project directory to 
"""

from sys import path
from constants import PROJECT_DIR, API_HOST, API_PORT
path.insert(1, PROJECT_DIR)

from master import api

if __name__ == "__main__":
    api.app.run(API_HOST, API_PORT, debug=True)