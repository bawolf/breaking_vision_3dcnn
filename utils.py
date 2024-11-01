import os
from datetime import datetime

def create_run_directory(base_dir='outputs'):
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a new directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    
    return run_dir