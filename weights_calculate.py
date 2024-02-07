import numpy as np

# Given the importance order: match, set, game, point
# We will construct a pairwise comparison matrix for AHP
# For simplicity, we will assign values based on the order of importance
# These values are often determined through expert judgment or consensus
# Here we use an arbitrary scale where the next important factor is half as important as the previous one

# Pairwise comparison matrix
# match set  game point
A = np.array([[1,   2,   4,   8],  # match
              [0.5, 1,   2,   4],  # set
              [0.25,0.5, 1,   2],  # game
              [0.125,0.25,0.5, 1]])  # point

# Calculate the eigenvector and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

# Find the index of the maximum eigenvalue
max_index = np.argmax(eigenvalues)

# The corresponding eigenvector is the AHP weights
weights = eigenvectors[:, max_index].real

# Normalizing the weights
weights = weights / np.sum(weights)

print(weights)
