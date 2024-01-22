from sklearn.cluster import KMeans
from skimage.io import imread_collection
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

# Load images from a directory
images = imread_collection('YourRoot')

# Check if the images collection is not empty
if len(images) == 0:
    raise ValueError("No images found in the directory.")

# Convert images to grayscale and resize
processed_images = [resize(rgb2gray(image), (224, 224)) for image in images]

# Flatten the images for clustering
flattened_images = [image.reshape(-1) for image in processed_images]

# Stack images into a single numpy array
data = np.vstack(flattened_images)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data)

# Get the cluster labels
labels = kmeans.labels_

# Calculate the Davies-Bouldin index
db_index = davies_bouldin_score(data, labels)

# Print the Davies-Bouldin index
print(f"Davies-Bouldin Index: {db_index}")

# Function to display images with cluster label as title
def display_cluster(images, labels, cluster_num):
    # Filter images belonging to the specified cluster
    cluster_images = np.array(images)[np.where(labels == cluster_num)[0]]

    # Display first few images of the cluster
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(ax.flatten()):
        if i < len(cluster_images):
            ax.imshow(cluster_images[i],cmap='Accent')
            ax.axis('off')
        else:
            ax.set_visible(False)
    plt.suptitle(f"Cluster {cluster_num}")
    plt.show()

# Display images from each cluster
for cluster in range(kmeans.n_clusters):
    display_cluster(processed_images, labels, cluster)


# The range of cluster numbers to try
cluster_range = range(1, 10)

# List to store the inertia for each cluster number
inertias = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Plotting the Elbow graph
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, inertias, marker='o')
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.xticks(cluster_range)
plt.show()
