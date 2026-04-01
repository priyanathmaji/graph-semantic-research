import numpy as np
import torch

class create_npx:

    def create_npz_dataset(node_features_file: str, labels_file: str, edges_file: str, num_splits: str, output_file_name:str):
        """
        node_features_file = path of the csv file that contains the features for each node ex. features.csv
                            [0.000000,0.000000,1.000000,....,0.000000]
                            [0.000000,1.000000,0.000000,....,0.000000]
        labels_file = path of the csv file that contains the label for each node ex. labels.csv
                            [3]
                            [4]
        edges_file = path of the txt file that contains the edges [num_edges, 2] ex. edges.txt
                            [0 633]
                            [0 1862]
                            [0 2582]
                            [1 2]
                            [1 652]
        num_split = number of training / validation / test splits

        """

        node_features = np.loadtxt(node_features_file, delimiter=',', dtype=np.float32)
        node_labels = np.loadtxt(labels_file, delimiter=',', dtype=np.int64)
        edges = np.loadtxt(edges_file, dtype=np.int64)

        num_nodes = node_features.shape[0]
        train_masks = np.zeros((num_splits,num_nodes), dtype=bool)
        val_masks = np.zeros((num_splits,num_nodes), dtype=bool)
        test_masks = np.zeros((num_splits,num_nodes), dtype=bool)
        
        for i in range(num_splits):
            indices = np.arange(num_nodes)
            np.random.shuffle(indices)

            train_idx = indices[:int(0.6*num_nodes)]
            val_idx = indices[int(0.6*num_nodes): int(0.8*num_nodes)]
            test_idx = indices[int(0.8*num_nodes):]

            train_masks[i, train_idx] = True
            val_masks[i, val_idx] = True
            test_masks[i, test_idx] = True

        print(f"Verified Split 0 - Train: {train_masks[0].sum()}, Val: {val_masks[0].sum()}, Test: {test_masks[0].sum()}")
        
        print("no of nodes", num_nodes)
        print("no of features per node", node_features.shape[1])
        print("no of node labels, should matach no of nodes", node_labels.shape[0])
        print("no of edges", edges.shape[0])
        np.savez(output_file_name,
                node_features=node_features,
                node_labels=node_labels,
                edges=edges,
                train_masks=train_masks,
                val_masks=val_masks,
                test_masks=test_masks)



