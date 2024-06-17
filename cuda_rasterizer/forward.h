/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
    /**
     * 1. 计算3D covariance matrix
     * 2. 计算2D covariance matrix和它的inverse
     * 3. 计算投影以后的影响矩阵
     * 4. 从Spherical harmonic coefficients计算点的rgb
     * @param P               [R] number of points
     * @param D               [R] maximum spherical harmonic degree
     * @param M               [R] number of spherical coefficients
     * @param orig_points     [R] (P, 3) original point set
     * @param scales          [R] (P, 3)
     * @param scale_modifier  [R]
     * @param rotations       [R] (P, 4)
     * @param opacities       [R] (P, 1)
     * @param shs             [R] (P, 1, 3) spherical harmonic coefficients
     * @param clamped         [W]
     * @param cov3D_precomp   [R] (0) or ?
     * @param colors_precomp  [R] (0) or ?
     * @param viewmatrix      [R] (4, 4) 旋转矩阵，从world coordinate转化到camera coordinate
     * @param projmatrix      [R] (4, 4) 投影矩阵，从world coordinate转化到image plane上
     * @param cam_pos         [R] (3, )
     * @param W               [R] image width
     * @param H               [R] image height
     * @param focal_x         [R]
     * @param focal_y         [R]
     * @param tan_fovx        [R]
     * @param tan_fovy        [R]
     * @param radii           [W] 高斯投影在image plane影响的最大范围
     * @param points_xy_image [W] 点投影在image plane上的pixel index
     * @param depths          [W] 点在camera coordinate下的z值，表示深度信息，可以用来表达点与点的前后关系
     * @param cov3Ds          [W] 3D covariance matrix
     * @param colors          [W] 点的RGB
     * @param conic_opacity   [W] the inverse of 2D covariance matrix, opacity
     * @param grid            [W]
     * @param tiles_touched   [W]
     * @param prefiltered
     */
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float* depths,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_depth);
}


#endif