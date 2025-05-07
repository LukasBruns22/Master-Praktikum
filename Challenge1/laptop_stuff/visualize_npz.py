import numpy as np

# Load the .npz file
data = np.load('1.npz')

# List all the keys (which correspond to the variables stored in the file)
print(data.files)

# Optionally, print some of the arrays to understand their structure
for key in data.files:
    print(f"{key}: {data[key].shape}")
    # Uncomment this to print the data, but it might be large
    # print(data[key])
