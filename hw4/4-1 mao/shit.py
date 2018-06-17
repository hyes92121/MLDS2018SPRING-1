import numpy as np
import matplotlib.pyplot as plt

data = np.zeros( (512,512,3), dtype=np.uint8)
data[256,256] = [255,0,0]
plt.imshow(data, interpolation='nearest')
plt.show()