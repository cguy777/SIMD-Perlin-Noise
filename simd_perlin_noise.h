/*
Copyright (c) 2026, Noah McLean.
This is licensed with the 3-clause BSD license.
See the license info at the bottom of this file.

********** Basic Info and Attribution **********

This is a simple 2D simplex perlin noise generator implemented in SIMD using SSE 4.1.
It is based off of an implementation by Sebastien Rombauts (sebastien.rombauts@gmail.com).
Because this library is essentially a derivation, I feel obligated to point you towards the
original library, which can be found here: https://github.com/SRombauts/SimplexNoise
The original is distributed under the MIT license and can be found at the bottom of this file.
I also referred to the paper "Simplex noise demystified" by Stefan Gustavson, which the original
library was based from: https://www.researchgate.net/publication/216813608_Simplex_noise_demystified

********** API USAGE **********

This is a single file library in the style of the STB libraries. You may include this file
anywhere, but somewhere (and only once) you MUST define SIMD_PERLIN_NOISE_IMPLEMENTATION 
before the #include for this file, which will create the function implementations.
Typical usage is as follows:

PerlinSimplexFractalParams params = PerlinSimplexGetDefaultParams();
float x = 0.1f;
float y = 0.2f;
float noiseValue = PerlinSimplexFractal(x, y, params);

You can also just get direct 2D samples without using the fractal/fractional function by
calling the noise functions directly. The SIMD and scalar noise function signatures are below:

__m128 PerlinSimplexNoise(__m128 x_128, __m128 y_128, int32_t seed);
float PerlinSimplexNoiseScalar(float x, float y, int32_t seed);

Obviously, you can change the parameters as necessary.

********** More Info **********

The main thing I did was take the 2D version present in the original version and
reimplement it using SIMD. The SIMD implementation is across multiple passes/octaves.
The benefit appears when using more than 2 passes, in which the speed increase becomes
evident. At 8 passes, this implementation is about twice as fast.  At 12 passes you
should see speeds that are around 3 times faster than the original scalar implementation.
This was measured using MSVC and O2 on a i3-10100. Because this is a near 1-1 translation
of a scalar version, I won't claim that this is a particularly good implementation.
However, it seems to work relatively well, especially when the pass count is in multiples
of four. I also switched from the 1-byte hash table to a hash function as it was simpler to
implement in SIMD. The hash behavior seems to be good enough from what I've seen, but you
may want to reach for a different one or reimplement the hash table version again. I also
added a seed parameter to help add some randomness if desired. The other change made was
switching back to a table-based vector approach. This is faster and seems work better
from a noise generation standpoint.
*/

#pragma once

#include <stdint.h>
#include <intrin.h>

typedef struct {
    /**
    The frequency of the first pass.
    You'll usually want to set this to 1.0
    */
    float initialFreq;
    
    /**
    The amplitude range of the first pass.
    You'll usually want to set this to 1.0
    */
    float initialAmplitude;
    
    /**
    This number is multiplied by the frequency per pass to determine the frequency of the next pass.
    2.0 is default, meaning that the frequency doubles on each successive pass. (lacunarity)
    */
    float freqChangeFactor;
    
    /**
    This number is multiplied by the amplitude per pass to determine the amplitude of the next pass.
    0.5 is the default, meaning the amplitude range is halved on each successive pass. (persistence)
    */
    float ampChangeFactor;
    
    /**
    The amount of passes per noise sample (octaves).
    */
    int32_t passCount;
    
    /**
    This is an offset that is provided to the hash function to help provide some randomness if desired.
    The default is 0.
    On each successive pass, the seed is incremented by one to further add some randomness.
    */
    int32_t seed;
} PerlinSimplexFractalParams;

/**
Returns a default set of parameters
*/
PerlinSimplexFractalParams PerlinSimplexGetDefaultParams();

/**
A SIMD version of a hash found here:
https://www.burtleburtle.net/bob/hash/integer.html
*/
__m128i PerlinHash(__m128i i, int32_t seed);

/**
Compute gradients-dot-residual vectors (2D).
I'm not a huge math guy, so I'm copying the language from the original scalar version and assuming it's correct.
*/
__m128 PerlinGrad(__m128i hash_128, __m128 x_128, __m128 y_128);

/**
SIMD 2D Perlin simplex noise.
Takes sets of 2D coordinates and a seed.
Returns a noise value between -1 and 1.
A value of 0 is returned for all whole-number coordinates.
*/
__m128 PerlinSimplexNoise(__m128 x_128, __m128 y_128, int32_t seed);

/**
Convenience function to call the SIMD version of PerlinSimplexNoise in a scalar way.
*/
float PerlinSimplexNoiseScalar(float x, float y, int32_t seed);

/**
From original:
"Fractal/Fractional Brownian Motion (fBm) summation of 2D Perlin Simplex noise"
Takes a 2D coordinate and parameters and returns a summed noise value from that point.
See the PerlinNoiseParams struct for more info on the parameters.
The returned value is normalized between -1 and 1.
*/
float PerlinSimplexFractal(float x, float y, PerlinSimplexFractalParams params);

#ifdef SIMD_PERLIN_NOISE_IMPLEMENTATION
//Implementation below

const PerlinSimplexFractalParams DEFAULT_SIMPLEX_PARAMS = {
    1.0f,
    1.0f,
    2.0f,
    0.5f,
    8,
    0
};

PerlinSimplexFractalParams PerlinSimplexGetDefaultParams() {
    return DEFAULT_SIMPLEX_PARAMS;
}

__m128i PerlinHash(__m128i i, __m128i seed) {
    //SIMD version of a hash found here:
    //https://www.burtleburtle.net/bob/hash/integer.html
    i = _mm_add_epi32(i, seed);
    i = _mm_xor_si128(i, _mm_srai_epi32(i, 4));
    __m128i a_xor_deadbeef = _mm_xor_si128(i, _mm_set1_epi32(0xdeadbeef));
    __m128i i_shift_left_5 = _mm_slli_epi32(i, 5);
    i = _mm_add_epi32(a_xor_deadbeef, i_shift_left_5);
    i = _mm_xor_si128(i, _mm_srai_epi32(i, 11));
    return i;
}

const float PerlinSimplexGradTable[8][2] = {
    {1, 0}, {-1, 0}, {0, 1},   {0, -1},
    {0, 1}, {-1, 1}, {-1, -1}, {1, -1}
};

__m128 PerlinGrad(__m128i hash_128, __m128 x_128, __m128 y_128) {
    __m128i h_128i = _mm_and_si128(hash_128, _mm_set1_epi32(0b111)); // Convert low 3 bits of hash code
    
    //Driving this from a table of 2D vectors seems to give better noise generation
    //than the heavy branching version in the original
    //TODO: is there a more SIMD way to assign the vectors here?
    __m128 grad_table_x_128 = _mm_set_ps(PerlinSimplexGradTable[((uint32_t*) &h_128i)[3]][0],
                                         PerlinSimplexGradTable[((uint32_t*) &h_128i)[2]][0],
                                         PerlinSimplexGradTable[((uint32_t*) &h_128i)[1]][0],
                                         PerlinSimplexGradTable[((uint32_t*) &h_128i)[0]][0]);
                                        
    __m128 grad_table_y_128 = _mm_set_ps(PerlinSimplexGradTable[((uint32_t*) &h_128i)[3]][1],
                                         PerlinSimplexGradTable[((uint32_t*) &h_128i)[2]][1],
                                         PerlinSimplexGradTable[((uint32_t*) &h_128i)[1]][1],
                                         PerlinSimplexGradTable[((uint32_t*) &h_128i)[0]][1]);
    
    __m128 x_mul_128 = _mm_mul_ps(x_128, grad_table_x_128);
    __m128 y_mul_128 = _mm_mul_ps(y_128, grad_table_y_128);
    
    __m128 component_wise_sum_128 = _mm_add_ps(x_mul_128, y_mul_128);
    return component_wise_sum_128;
}

//2D perlin simplex noise in simd
__m128 PerlinSimplexNoise(__m128 x_128, __m128 y_128, __m128i seed) {
    
    // Skewing/Unskewing factors for 2D
    __m128 F2_128 = _mm_set1_ps(0.366025403f); // F2 = (sqrt(3) - 1) / 2
    __m128 G2_128 = _mm_set1_ps(0.211324865f); // G2 = (3 - sqrt(3)) / 6   = F2 / (1 + 2 * K)
    
    // Skew the input space to determine which simplex cell we're in
    __m128 s_128 = _mm_mul_ps(_mm_add_ps(x_128, y_128), F2_128); // Hairy factor for 2D
    __m128 xs_128 = _mm_add_ps(x_128, s_128);
    __m128 ys_128 = _mm_add_ps(y_128, s_128);
    //These floors are the only SSE 4.1 intrinsics.
    //Everything else except for the horizontal adds are SSE 1 and 2.
    __m128 i_128 = _mm_floor_ps(xs_128);
    __m128 j_128 = _mm_floor_ps(ys_128);
    
    // Unskew the cell origin back to (x,y) space
    __m128 t_128 = _mm_mul_ps(_mm_add_ps(i_128, j_128), G2_128);
    __m128 x0_128 = _mm_sub_ps(x_128, _mm_sub_ps(i_128, t_128));
    __m128 y0_128 = _mm_sub_ps(y_128, _mm_sub_ps(j_128, t_128));
    
    // For the 2D case, the simplex shape is an equilateral triangle.
    // Determine which simplex we are in.
    __m128 i1_128 = _mm_set1_ps(1.0f);
    __m128 j1_128 = _mm_set1_ps(1.0f);
    __m128 cmpMaskGt = _mm_cmpgt_ps(x0_128, y0_128);
    //Do we need this?
    __m128 cmpMaskNgt = _mm_cmpngt_ps(x0_128, y0_128);
    
    i1_128 = _mm_and_ps(i1_128, cmpMaskGt);
    //Can we use the first mask with a and-not intrinsic here?
    j1_128 = _mm_and_ps(j1_128, cmpMaskNgt);
    
    // A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    // a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    // c = (3-sqrt(3))/6
    
    __m128 x1_128 = _mm_add_ps(_mm_sub_ps(x0_128, i1_128), G2_128);
    __m128 y1_128 = _mm_add_ps(_mm_sub_ps(y0_128, j1_128), G2_128);
    
    __m128 G2_128Times2 = _mm_mul_ps(G2_128, _mm_set1_ps(2.0f));
    
    __m128 x2_128 = _mm_add_ps(_mm_sub_ps(x0_128, _mm_set1_ps(1.0f)), G2_128Times2);
    __m128 y2_128 = _mm_add_ps(_mm_sub_ps(y0_128, _mm_set1_ps(1.0f)), G2_128Times2);
    
    // Work out the hashed gradient indices of the three simplex corners
    __m128i i_128i = _mm_cvtps_epi32(i_128);
    __m128i j_128i = _mm_cvtps_epi32(j_128);
    __m128i gi0_128i = PerlinHash(_mm_add_epi32(i_128i, PerlinHash(j_128i, seed)), seed);
    
    __m128i j_plus_j1_hash128i = PerlinHash(_mm_cvtps_epi32(_mm_add_ps(j_128, j1_128)), seed);
    __m128i gi1_128i = PerlinHash(_mm_add_epi32(i_128i, _mm_add_epi32(_mm_cvtps_epi32(i1_128), j_plus_j1_hash128i)), seed);
    
    __m128i one_128i = _mm_set1_epi32(1);
    __m128i j_plus_one_hash128i = PerlinHash(_mm_add_epi32(j_128i, one_128i), seed);
    __m128i gi2_128i = PerlinHash(_mm_add_epi32(i_128i, _mm_add_epi32(one_128i, j_plus_one_hash128i)), seed);
    
    //We use these values a few times, so we'll just create them here
    __m128 half_128 = _mm_set1_ps(0.5f);
    __m128 zero_128 = _mm_setzero_ps();
    
    // Calculate the contribution from the first corner
    __m128 x0_squared_128 = _mm_mul_ps(x0_128, x0_128);
    __m128 y0_squared_128 = _mm_mul_ps(y0_128, y0_128);
    __m128 t0_128 = _mm_sub_ps(_mm_sub_ps(half_128, x0_squared_128), y0_squared_128);
    
    __m128 t0_lt_zero_mask = _mm_cmpnlt_ps(t0_128, zero_128);
    t0_128 = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(t0_128, t0_128), t0_128), t0_128);
    __m128 n0_128 = _mm_mul_ps(t0_128, PerlinGrad(gi0_128i, x0_128, y0_128));
    n0_128 = _mm_and_ps(n0_128, t0_lt_zero_mask);
    
    // Calculate the contribution from the second corner
    __m128 x1_squared_128 = _mm_mul_ps(x1_128, x1_128);
    __m128 y1_squared_128 = _mm_mul_ps(y1_128, y1_128);
    __m128 t1_128 = _mm_sub_ps(_mm_sub_ps(half_128, x1_squared_128), y1_squared_128);
    
    __m128 t1_lt_zero_mask = _mm_cmpnlt_ps(t1_128, zero_128);
    t1_128 = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(t1_128, t1_128), t1_128), t1_128);
    __m128 n1_128 = _mm_mul_ps(t1_128, PerlinGrad(gi1_128i, x1_128, y1_128));
    n1_128 = _mm_and_ps(n1_128, t1_lt_zero_mask);
    
    // Calculate the contribution from the third corner
    __m128 x2_squared_128 = _mm_mul_ps(x2_128, x2_128);
    __m128 y2_squared_128 = _mm_mul_ps(y2_128, y2_128);
    __m128 t2_128 = _mm_sub_ps(_mm_sub_ps(half_128, x2_squared_128), y2_squared_128);
    
    __m128 t2_lt_zero_mask = _mm_cmpnlt_ps(t2_128, zero_128);
    t2_128 = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(t2_128, t2_128), t2_128), t2_128);
    __m128 n2_128 = _mm_mul_ps(t2_128, PerlinGrad(gi2_128i, x2_128, y2_128));
    n2_128 = _mm_and_ps(n2_128, t2_lt_zero_mask);
    
    // Add contributions from each corner to get the final noise value.
    // The result is scaled to return values in the interval [-1,1].
    __m128 output128 = _mm_add_ps(n0_128, _mm_add_ps(n1_128, n2_128));
    output128 = _mm_mul_ps(output128, _mm_set1_ps(70.0f));
    return output128;
}

//Convenience function to call the SIMD version of PerlinSimplexNoise in a scalar way.
float PerlinSimplexNoiseScalar(float x, float y, int32_t seed) {
    __m128 x_128 = _mm_set1_ps(x);
    __m128 y_128 = _mm_set1_ps(y);
    __m128i seed_128 = _mm_set1_epi32(seed);
    __m128 noise_128 = PerlinSimplexNoise(x_128, y_128, seed_128);
    float noise = _mm_cvtss_f32(noise_128);
    return noise;
}

//Precompute some output masks and then grab them from a table
const __m128 perlin_output_mask_0_128 = _mm_setzero_ps();
const __m128 perlin_output_mask_1_128 = _mm_castsi128_ps(_mm_set_epi32(0,          0,          0,          0xFFFFFFFF));
const __m128 perlin_output_mask_2_128 = _mm_castsi128_ps(_mm_set_epi32(0,          0,          0xFFFFFFFF, 0xFFFFFFFF));
const __m128 perlin_output_mask_3_128 = _mm_castsi128_ps(_mm_set_epi32(0,          0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));
const __m128 perlin_output_mask_4_128 = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));
const __m128 perlin_output_masks[] = {
    perlin_output_mask_0_128,
    perlin_output_mask_1_128,
    perlin_output_mask_2_128,
    perlin_output_mask_3_128,
    perlin_output_mask_4_128
};

const __m128i perlin_four_pass_seed_offset_128i = _mm_set_epi32(3, 2, 1, 0);
const __m128i perlin_four_128i = _mm_set1_epi32(4);

float PerlinSimplexFractal(float x, float y, PerlinSimplexFractalParams params) {
    float currentFrequency = params.initialFreq;
    float currentAmplitude = params.initialAmplitude;
    float denom = 0.0f;
    
    __m128 x_128 = _mm_set1_ps(x);
    __m128 y_128 = _mm_set1_ps(y);

    __m128 output_128 = _mm_setzero_ps();
    __m128i passIndex_128i = _mm_setzero_si128();
    
    __m128i seed_128 = _mm_add_epi32(_mm_set1_epi32(params.seed), perlin_four_pass_seed_offset_128i);

    for(uint32_t passIndex = 0; passIndex < params.passCount; passIndex += 4) {
        __m128 fourPassOutput_128 = _mm_setzero_ps();
    
        __m128 currentFreq_128 = _mm_set1_ps(currentFrequency);
        __m128 currentAmp_128 = _mm_set1_ps(currentAmplitude);
        
        seed_128 = _mm_add_epi32(seed_128, passIndex_128i);
        
        uint32_t remainingPasses = params.passCount - passIndex;
        uint32_t iterCount = remainingPasses >= 4 ? 4 : remainingPasses;
        for(uint32_t i = 0; i < iterCount; i++) {
            ((float*) &currentFreq_128)[i] = currentFrequency;
            ((float*) &currentAmp_128)[i] = currentAmplitude;
            
            denom += currentAmplitude;
            currentFrequency *= params.freqChangeFactor;
            currentAmplitude *= params.ampChangeFactor;
        }
        
        __m128 freqAdjX_128 = _mm_mul_ps(x_128, currentFreq_128);
        __m128 freqAdjY_128 = _mm_mul_ps(y_128, currentFreq_128);
        
        fourPassOutput_128 = _mm_mul_ps(currentAmp_128, PerlinSimplexNoise(freqAdjX_128, freqAdjY_128, seed_128));
        
        //Mask the result so we get the desired amount of passes
        output_128 = _mm_add_ps(output_128, _mm_and_ps(fourPassOutput_128, perlin_output_masks[iterCount]));
        
        //We have to manually increment passIndex_128i
        passIndex_128i = _mm_add_epi32(passIndex_128i, perlin_four_128i);
    }
    
    //Sum all of the lanes together
    //These horizontal adds are SSE 3
    output_128 = _mm_hadd_ps(output_128, output_128);
    output_128 = _mm_hadd_ps(output_128, output_128);
    float output = _mm_cvtss_f32(output_128);
    
    return output / denom;
}

#endif // SIMD_PERLIN_NOISE_IMPLEMENTATION



//Licenses below

/*
License of this code contained in this file:

BSD 3-Clause License

Copyright (c) 2026, Noah McLean

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
License of the original:

The MIT License (MIT)

Copyright (c) 2012-2018 Sebastien Rombauts (sebastien.rombauts@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
