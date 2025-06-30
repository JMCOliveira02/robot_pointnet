import numpy as np

colors_segmentation = np.array([ 
        [255, 0, 0],       # 0 - ceiling      - red
        [0, 255, 0],       # 1 - floor        - green
        [0, 0, 255],       # 2 - wall         - blue
        [0, 0, 0]          # 3 - clutter      - black
    ])