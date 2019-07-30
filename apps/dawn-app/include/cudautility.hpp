/*
 * cudautility.hpp
 *
 *  Created on: May 14, 2019
 *      Author: Yanzhao Wu
 */

#ifndef INCLUDE_CUDAUTILITY_HPP_
#define INCLUDE_CUDAUTILITY_HPP_

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)  if(x != cudaSuccess) \
    PERR("error: cuda err=%s", cudaGetErrorString (cudaGetLastError()));
#endif

#ifndef CU_CHECK
#define CU_CHECK(x)  if(x != CUDA_SUCCESS ) \
    PERR("error: cuda err=%s", cudaGetErrorString (cudaGetLastError()));
#endif

#endif /* INCLUDE_CUDAUTILITY_HPP_ */
