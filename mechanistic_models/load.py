"""
Convenience script to read the model pickles from the optimization scripts
and print their best loss and the final parameters

@author: Lynn Schmittwilken, Feb 2025
"""

import pickle

results_file = "./results/active_multi.pickle"

if __name__ == "__main__":
    # Load data from pickle:
    with open(results_file, 'rb') as handle:
        data_pickle = pickle.load(handle)
    
    best_params = data_pickle["best_params_auto"]
    best_loss = data_pickle["best_loss_auto"]
    model_params = data_pickle["params_dict"]
    
    print("--------------------------------")
    print("Summary of " + results_file.split("/")[-1])
    print("--------------------------------")
    try:
        print("Gain:", data_pickle["model_params"]["gain"])
        print("Same noise instances:", data_pickle["model_params"]["sameNoise"])
    except:
        pass

    print("Best loaded loss:", best_loss)
    print("Best params", best_params)
    print("--------------------------------")
