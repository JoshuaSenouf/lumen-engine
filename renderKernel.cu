#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "cutil_math.h"
#include "device_launch_parameters.h"

#include "renderKernel.h"


void lumenRender(int width, int height, int samples, int bounces)
{
    printf("\nRENDER CONFIG :\n\n"
           "Width = %d\n"
           "Height = %d\n"
           "Samples = %d\n"
           "Bounces = %d\n\n"
           "//////////////////////\n", width, height, samples, bounces);
}
