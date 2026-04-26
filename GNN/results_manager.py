import json
import os
import pandas as pd

class ExperimentTracker:

    def __init__(self, dataset_name, node_feature_type, save_dir='results'):
        self.dataset_name = dataset_name
        self.node_feature_type = node_feature_type
        self.save_dir = save_dir
        # We now use a dictionary to store multiple histories: {model_name: [list of epochs]}
        self.histories = {} 
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def log_epoch(self, model_name, epoch, loss, val_f1, val_acc):
        """Pass the model name (e.g., 'gcn2') every time you log an epoch."""

        l_val = loss.item() if hasattr(loss, 'item') else loss

        if model_name not in self.histories:
            self.histories[model_name] = []
            
        self.histories[model_name].append({
        'epoch': epoch,
        'train_loss': l_val,
        'val_f1': val_f1,
        'val_acc': val_acc
        })

    def save_results(self, model_name, test_f1, test_acc, params=None):
        """Pass the model name when saving the final test results."""
        file_id = f"{self.dataset_name}_{self.node_feature_type}_{model_name}".lower().replace(" ", "_")
        
        clean_params = {}
        if params:
            for k, v in params.items():
                # Convert non-serializable objects (like device) to strings
                if hasattr(v, '__dict__') or "device" in str(type(v)).lower():
                    clean_params[k] = str(v)
                else:
                    clean_params[k] = v
        
        if model_name in self.histories:
            hist_df = pd.DataFrame(self.histories[model_name])
            hist_df.to_csv(os.path.join(self.save_dir, f"{file_id}_history.csv"), index=False)

        summary = {
            "dataset": self.dataset_name,
            "features": self.node_feature_type,
            "model": model_name,
            "test_f1": round(test_f1, 4),
            "test_acc": round(test_acc, 4),
            "params": clean_params or {}
        }
        
        master_path = os.path.join(self.save_dir, 'master_leaderboard.json')
        all_data = []
        if os.path.exists(master_path):
            with open(master_path, 'r') as f:
                all_data = json.load(f)
        
        all_data.append(summary)
        with open(master_path, 'w') as f:
            json.dump(all_data, f, indent=4)
        
        print(f"{model_name.upper()} results saved using {self.node_feature_type} features.")