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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

/**
 *
 * @param background    (3, )
 * @param means3D       (P, 3)
 * @param colors        (0) or ?
 * @param opacity       (P, 1)
 * @param scales        (P, 3)
 * @param rotations     (P, 4)
 * @param scale_modifier
 * @param cov3D_precomp (0) or ?
 * @param viewmatrix    (4, 4)
 * @param projmatrix    (4, 4)
 * @param tan_fovx
 * @param tan_fovy
 * @param image_height
 * @param image_width
 * @param sh            (P, 1, 3)
 * @param degree
 * @param campos        (3, )
 * @param prefiltered
 * @return
 */
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered);

/**
 *
 * @param background     (3, )
 * @param means3D        (P, 3)
 * @param radii          (P, )
 * @param colors         (0) or ?
 * @param scales         (P, 3)
 * @param rotations      (P, 4)
 * @param scale_modifier
 * @param cov3D_precomp  (0) or ?
 * @param viewmatrix     (4, 4)
 * @param projmatrix     (4, 4)
 * @param tan_fovx
 * @param tan_fovy
 * @param dL_dout_color  [W]
 * @param dL_dout_depth  [W]
 * @param sh             (P, 1, 3)
 * @param degree         ()
 * @param campos         (3, )
 * @param geomBuffer
 * @param R              num_rendered
 * @param binningBuffer
 * @param imageBuffer
 * @return
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);