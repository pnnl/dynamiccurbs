import pickle
import numpy as np
from kmodes.kmodes import KModes
from kmodes.util import dissim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_configs(model_path, integer_to_color_map, integer_to_space_type_map):
    #this only works for even k since this plots in 2 columns
    all_labels = list(integer_to_space_type_map.keys())
    shown_labels = set()
    
    with open(model_path, 'rb') as d:
        km = pickle.load(d)[0]
    k_star = km.cluster_centroids_.shape[0]
    
    fig, axs = plt.subplots(int(k_star/2), 2, figsize=(20,8))#, dpi=300)
    x_pos = np.arange(0,100,1)
    height = np.ones((100,))
    
    counter = 0
    
    for index, val in np.ndenumerate(np.ones((int(k_star/2),2))):
        axs[index].bar(x_pos, height, width=1.0, align="center", color=[ integer_to_color_map[k] for k in km.cluster_centroids_[counter,:] ])
        labels = list(set(km.cluster_centroids_[counter,:]))
        for label in labels: shown_labels.add(label)
        if counter == k_star-1:
            handles = [plt.Rectangle((0,0),1,1, color=integer_to_color_map[label]) for label in shown_labels ]
            plt.legend(handles, [integer_to_space_type_map[i] for i in shown_labels ], loc='center left', bbox_to_anchor=(1.05, 4.5), fontsize=18)
        axs[index].label_outer()
        axs[index].set_ylabel("Mode " + str(counter+1), fontsize=18, rotation=90)
        axs[index].set_yticklabels([])
        axs[index].set_xlim([0,100])
        
        if counter == k_star - 1 or counter == k_star - 2:
            axs[index].set_xlabel("Proportion of blockface (ordered)", fontsize=20)
            axs[index].tick_params(axis='x', labelsize=18)
        counter += int(val)
        
        
    gs1 = gridspec.GridSpec(int(k_star/2), 2)
    gs1.update(wspace=0.1, hspace=0.1)
    
    plt.show()
    
def plot_configs_with_membership(model_path, integer_to_color_map, integer_to_space_type_map, total_num, membership_counts):
    #this only works for even k since this plots in 2 columns
    all_labels = list(integer_to_space_type_map.keys())
    shown_labels = set()
    
    with open(model_path, 'rb') as d:
        km = pickle.load(d)[0]
    k_star = km.cluster_centroids_.shape[0]
    
    fig, axs = plt.subplots(int(k_star/2), 2, figsize=(20,8))#, dpi=300)
    x_pos = np.arange(0,100,1)
    height = np.ones((100,))
    
    counter = 0
    
    for index, val in np.ndenumerate(np.ones((int(k_star/2),2))):
        axs[index].bar(x_pos, height, width=1.0, align="center", color=[ integer_to_color_map[k] for k in km.cluster_centroids_[counter,:] ])
        labels = list(set(km.cluster_centroids_[counter,:]))
        for label in labels: shown_labels.add(label)
        if counter == k_star-1:
            handles = [plt.Rectangle((0,0),1,1, color=integer_to_color_map[label]) for label in shown_labels ]
            plt.legend(handles, [integer_to_space_type_map[i] for i in shown_labels ], loc='center left', bbox_to_anchor=(1.05, 4.5), fontsize=18)
        axs[index].label_outer()
        axs[index].set_ylabel("Mode " + str(counter+1) + "\n" + "(" + str(membership_counts[counter])+ "/" + str(total_num) + ")", fontsize=18, rotation=90)
        axs[index].set_yticklabels([])
        axs[index].set_xlim([0,100])
        
        if counter == k_star - 1 or counter == k_star - 2:
            axs[index].set_xlabel("Proportion of blockface (ordered)", fontsize=20)
            axs[index].tick_params(axis='x', labelsize=18)
        counter += int(val)
        
        
    gs1 = gridspec.GridSpec(int(k_star/2), 2)
    gs1.update(wspace=0.1, hspace=0.1)
    
    plt.show()
    
def kmodes_clustering_loop_k(k_max, ekey_to_block_alloc, dissim_func, ekeys=None):
    all_data = []
    if ekeys == None:
        ekeys = ekey_to_block_alloc.keys()
    for ekey in ekeys:
        all_data.append(ekey_to_block_alloc[ekey])
    all_data = np.asarray(all_data)
    print(all_data.shape[0], " total samples")

    val_inds = np.random.choice(all_data.shape[0], size=int(0.1*all_data.shape[0]), replace=False) #hold out 10% of the data for cross validation
    train_inds = np.asarray([i for i in range(all_data.shape[0]) if i not in val_inds]) #use the rest for training

    train_data = all_data[train_inds,:]
    val_data = all_data[val_inds,:]

    models = []
    counter = 0

    for i in range(2,k_max):
        print("Training model for ", str(i), "clusters")

        #initialize and train the model
        km = KModes(n_clusters=i, init='Cao', n_init=5, max_iter=100, verbose=0, cat_dissim=dissim_func)
        clusters = km.fit_predict(train_data)
        models.append([km])

        #get the predicted clusters of the held out training and validation data
        val_assignments = np.asarray(km.predict(val_data))
        train_assignments = np.asarray(km.predict(train_data))

        val_var_dists = []
        train_var_dists = []
        for j in range(i):
            #inds = np.where(np.any(assignments==0, axis=0))
            val_inds = np.argwhere(val_assignments==j)[:,0]
            train_inds = np.argwhere(train_assignments==j)[:,0]

            #compute within cluster variance with dissim.matching_dissim for training and validation data, when validation data distance variance diverges as a function k, we know k is too large and the model has overfit
            centroid = km.cluster_centroids_[j]
            train_dist = dissim.matching_dissim(train_data[train_inds,:], centroid) #compute the distances of all the training samples
            train_var_dist = np.var(train_dist) #compute the variance of these distances
            train_var_dists.append(train_var_dist) #save with model

            val_dist = dissim.matching_dissim(val_data[val_inds,:], centroid) #compute the distances of all the validation samples
            val_var_dist = np.var(val_dist) #compute the variance of these distances
            val_var_dists.append(val_var_dist) #save with model

        models[counter].append(np.mean(train_var_dists))
        models[counter].append(np.mean(val_var_dists))

        counter+=1 #increment counter for saving models and validation score
        km = None #reset model variable
    
    return(models)