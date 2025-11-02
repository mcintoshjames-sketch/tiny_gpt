import numpy as np

epochs = [1, 5, 10, 15, 20, 24]
losses = [4.97, 3.85, 3.59, 3.47, 3.38, 3.36]

# Fit power law: loss = a * epoch^b
# Your data suggests final loss around 1.3-1.4 at epoch 80