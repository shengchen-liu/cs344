// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //DONE
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion

  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset

  // for example, threadIdx: [1, 2, 1], blockIdx:[0, 1, 1], blockDim: [32, 32, 1]
  // a block can be split into mutliple threads

  // int y = threadIdx.y+ blockIdx.y* blockDim.y;
  // int x = threadIdx.x+ blockIdx.x* blockDim.x;

  // calculate 1d index location in rgbaImage

  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

  uchar4 rgba = rgbaImage[threadId];
  float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
  greyImage[threadId] = (unsigned char) channelSum;

  // // calculate 1d index location in rgbaImage
  // if (y < numCols && x < numRows) {
  //   int index = numCols * x + y;
  //   uchar4 rgba = rgbaImage[index];
  //   float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
  //   greyImage[index] = (unsigned char) channelSum;
  // }
    

}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched

  // set blockWidth
  int blockWidth = 32;

  const dim3 blockSize(blockWidth, blockWidth, 1);  //DONE

  // calculate grid size
  int bx = numRows / blockWidth + 1;
  int by = numCols / blockWidth + 1;
  // bz = 1
  const dim3 gridSize( bx, by, 1);  //DONE

  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
