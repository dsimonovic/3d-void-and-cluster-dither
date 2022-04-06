# Generating 3D dithering pattern (blue noise) 

Algorithm is verbatim generalization of the R. Ulichney (1993) void and cluster algorithm for 2D. The main difference is the use of 3D kernel.

- R. Ulichney, “The Void-and-Cluster Method for Generating Dither Arrays”,Human Vision, Visual Processing, and Digital Display IV, J. Allebach and B. Rogowitz, eds., Proc. SPIE 1913, pp. 332-343, 1993.

Author's version of the paper can be found at: http://cv.ulichney.com/papers/1993-void-cluster.pdf

This is a single C++ file, requires OpenCV and at least C++11

INPUT:  Modifiable parameters are at the top of the cpp file.

OUTPUT: 3D pixel matrix is saved as a number of images (layers).

Friendly note: Using a single layer of a 3D dither pattern is inferior to 2D dither pattern (where inferior means low frequency noise).
