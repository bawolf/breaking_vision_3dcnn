from train import train_model
import os
from datetime import datetime
import json
from pathlib import Path

def create_hyperparam_directory():
    """Create a parent directory for hyperparameter search results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "outputs"
    hyperparam_dir = os.path.join(base_dir, f"run_{timestamp}_hyperparam")
    os.makedirs(hyperparam_dir, exist_ok=True)
    return hyperparam_dir

def create_run_directory_with_label(parent_dir, dataset_root):
    """Create a run directory with timestamp and dataset label"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract last two parts of dataset path for label
    path_parts = Path(dataset_root).parts
    dataset_label = '_'.join(path_parts[-2:])
    
    run_dir = os.path.join(parent_dir, f"run_{timestamp}_{dataset_label}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def run_hyperparameter_search(configs, dataset_roots):
    """
    Run hyperparameter search across multiple configurations and datasets
    
    Args:
        configs (list): List of configuration dictionaries
        dataset_roots (list): List of paths to different dataset roots
    """
    # Create parent directory for all runs
    hyperparam_dir = create_hyperparam_directory()
    
    # Save search parameters
    search_config = {
        'configs': configs,
        'dataset_roots': dataset_roots,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(hyperparam_dir, 'search_config.json'), 'w') as f:
        json.dump(search_config, f, indent=4)
    
    # Initialize results tracking
    all_results = []
    
    # Create a summary file
    summary_path = os.path.join(hyperparam_dir, 'results_summary.csv')
    with open(summary_path, 'w') as f:
        f.write("dataset,config_id,batch_size,learning_rate,best_val_loss,final_val_accuracy,early_stopped,run_dir\n")
    
    # Run each configuration on each dataset
    for dataset_idx, dataset_root in enumerate(dataset_roots):
        print(f"\nProcessing dataset {dataset_idx + 1}/{len(dataset_roots)}: {dataset_root}")
        
        for config_idx, config in enumerate(configs):
            print(f"\nRunning configuration {config_idx + 1}/{len(configs)}")
            print(f"Config: {config}")
            
            # Create run directory with dataset label
            run_dir = create_run_directory_with_label(hyperparam_dir, dataset_root)
            
            try:
                # Run training
                metrics = train_model(run_dir, dataset_root, config)
                
                # Store results
                result = {
                    'dataset_root': dataset_root,
                    'config': config,
                    'metrics': metrics,
                    'run_dir': run_dir
                }
                all_results.append(result)
                
                # Append to summary file
                with open(summary_path, 'a') as f:
                    f.write(f"{dataset_root},{config_idx},"
                           f"{config['batch_size']},{config['learning_rate']},"
                           f"{metrics['best_val_loss']:.4f},{metrics['final_val_accuracy']:.2f},"
                           f"{metrics['early_stopped']},{run_dir}\n")
                
                # Print results
                print(f"\nResults for dataset: {dataset_root}")
                print(f"Config: {config}")
                print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
                print(f"Final validation accuracy: {metrics['final_val_accuracy']:.2f}%")
                print(f"Run directory: {run_dir}")
                
            except Exception as e:
                print(f"Error running configuration on {dataset_root}: {str(e)}")
                # Log the error
                with open(os.path.join(hyperparam_dir, 'errors.log'), 'a') as f:
                    f.write(f"\nError in {run_dir}:\n{str(e)}\n")
    
    return hyperparam_dir, all_results

if __name__ == "__main__":
    # Example configurations
    configs = [
         {
            "batch_size": 8,
            "learning_rate": 0.001,
            "num_epochs": 50,
            "patience": 8
        },
        {
            "batch_size": 16,
            "learning_rate": 0.001,
            "num_epochs": 50,
            "patience": 8
        },
        {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 50,
            "patience": 8
        },
        {
            "batch_size": 64,
            "learning_rate": 0.001,
            "num_epochs": 50,
            "patience": 10
        }
    ]
    
    # Example dataset roots
    dataset_roots = [
        "../finetune/blog/bryant/random",
        "../finetune/blog/bryant/adjusted",
        "../finetune/blog/youtube/random",
        "../finetune/blog/youtube/adjusted",
        "../finetune/blog/combined/random",
        "../finetune/blog/combined/adjusted",
        "../finetune/blog/bryant_train_youtube_val/default"
    ]
    
    # Run hyperparameter search
    hyperparam_dir, results = run_hyperparameter_search(configs, dataset_roots)
    
    print(f"\nHyperparameter search complete!")
    print(f"Results are saved in: {hyperparam_dir}") 
