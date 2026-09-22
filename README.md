# SIMD-Perlin-Noise
SIMD 2D Simplex Perlin Noise implemented using SSE intrinsics.
Usage and documentation is at the top of the header file.
This is essentially a SIMD re-implementation of the 2D simplex noise implementation found here: [SimplexNoise](https://github.com/SRombauts/SimplexNoise]).
In almost all situations, the fractal noise function in this implementation is faster than the previously referenced library.
On the high end, this version is as much as 3-times faster.
The only time when I have recorded this implementation to be slower is with unoptimized builds or fractal pass/octave counts of less than three.
This library is licensed with the 3-clause BSD license.
Copyright (c) 2026, Noah McLean
