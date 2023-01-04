from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist


df_flow = pd.read_csv("seatac_vms1_vms2_treatment_ctl.csv")
df_flow['timestamp'] = pd.to_datetime(df_flow['timestamp'])
time_vector = df_flow['timestamp']


df_flow['day_of_week'] = df_flow['timestamp'].dt.day_of_week
df_flow['hour_of_day'] = df_flow['timestamp'].dt.hour
df_flow = df_flow[(df_flow.arr_crit_ratio > 0.2) & (df_flow.dep_crit_ratio > 0.2)]
# data_df = df_flow[['arr_crit_ratio', 'dep_crit_ratio']]
data_df = df_flow[['arr_crit_ratio', 'dep_crit_ratio', 'hour_of_day']]


###########################
# Compute optimal k;
# Elbow method: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

X = np.array(list(zip(data_df.values))).reshape(len(data_df), len(data_df.columns))
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 25)

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_


# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

##############################################

# Declaring Model
n_clusters = 6
model = KMeans(n_clusters=n_clusters, random_state=30)

# Fitting Model
kmeans = model.fit(data_df)

centroids = kmeans.cluster_centers_
print(centroids)

df_flow['cluster_labels'] = model.labels_
df_flow.to_csv('seatac_data_clustered.csv', index=False )

#####################################################
# Plotting

palette = sns.color_palette("colorblind", n_clusters)
colors = {k: v for k,v in zip(range(n_clusters), palette)}

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)

fig = plt.figure(figsize = (10, 7))

## For 3D plotting
ax = plt.axes(projection ="3d")
sctt = ax.scatter3D(df_flow['hour_of_day'], df_flow['dep_crit_ratio'], df_flow['arr_crit_ratio'], c=[palette[i] for i in kmeans.labels_], s=50, alpha=0.5)
ax.scatter3D(centroids[:, 2], centroids[:, 0], centroids[:, 1], c='red', s=50)


# # For 2D plotting
# sctt = plt.scatter(df_flow['arr_crit_ratio'], df_flow['dep_crit_ratio'], c= [palette[i] for i in kmeans.labels_], s=50, alpha=0.5)
# plt.scatter(centroids[:, 0],centroids[:, 1], c='red', s=50)
#
# for i, label in enumerate(df_flow['cluster_labels'].unique()):
#     plt.annotate(label,
#                  (centroids[i, 0], centroids[i,1]),
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  size=25, weight='bold',
#                  color='black')
# plt.xlabel('Arr. critical ratio', fontweight ='bold')
# plt.ylabel('Dep. critical ratio', fontweight ='bold')
#
# plt.show()

# fig.savefig('crit_ratio_clustered.png', bbox_inches='tight')