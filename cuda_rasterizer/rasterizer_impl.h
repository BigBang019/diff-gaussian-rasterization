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

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;        // size of temp space
		float* depths;           // 点在camera coordinate下的z值，表示深度信息，可以用来表达点与点的前后关系
		char* scanning_space;    // temp space
		bool* clamped;           // 标记哪些点的颜色<0
		int* internal_radii;
		float2* means2D;         // 点投影在image plane上的pixel index
		float* cov3D;            // 3D covariance matrix
		float4* conic_opacity;   // the inverse of 2D covariance matrix, opacity
		float* rgb;              // 点的rgb
		uint32_t* point_offsets; // 影响block数目的累加和，也可以理解成点-block的影响关系图（二分图），存在多少个边
		uint32_t* tiles_touched; // 点到底影响了多少个block

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;        // idx为block id，记录当前block被point_list_keys的哪些点影响：[x,y)
		uint32_t* n_contrib;  //
		float* accum_alpha;   //

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;                // size of temp space
		uint64_t* point_list_keys_unsorted; // keys: 高32bit标记当前点（point）影响哪个block，低32bit标记当前点的depth信息
		uint64_t* point_list_keys;          // ascending排序过后的keys
		uint32_t* point_list_unsorted;      // values：标记当前点（point）的idx
		uint32_t* point_list;               // 排序过后的corresponding values
		char* list_sorting_space;           // temp space

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};