# Generating 3D dithering pattern (blue noise) 

Algorithm is verbatim generalization of the R. Ulichney (1993) void and cluster algorithm for 2D. The main difference is the use of 3D kernel.

- R. Ulichney, “The Void-and-Cluster Method for Generating Dither Arrays”,Human Vision, Visual Processing, and Digital Display IV, J. Allebach and B. Rogowitz, eds., Proc. SPIE 1913, pp. 332-343, 1993.

Author's version of the paper can be found at: http://cv.ulichney.com/papers/1993-void-cluster.pdf

This is a single C++ file, requires OpenCV and at least C++11

INPUT:  Modifiable parameters are at the top of the cpp file.

OUTPUT: 3D pixel matrix is saved as a number of images (layers).

Kernel is Gaussian. Results tend to look more pleasing at lower end (1.3 - 1.4) which is in contrast to findings for 2D where sigma=1.9 is optimal:
- https://blog.demofox.org/2019/06/25/generating-blue-noise-textures-with-void-and-cluster/ and
- http://momentsingraphics.de/BlueNoise.html,

Friendly note: Using a single 2D layer out of a 3D dither pattern is inferior to a blue-noise 2D dither pattern (where inferior means low frequency noise). Please see: https://momentsingraphics.de/3DBlueNoise.html

Time complexity: O(*n∙m*∙log*n*), where *n* is a total number of pixels (voxels) and *m* is total filter size (in voxels)
