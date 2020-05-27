#pragma once

#include <cstring>
#include "typedefs.h"
#include "GL/glew.h"
#include "cuda_gl_interop.h"
#include "cuda_error_check.h"

class CudaGLBuffer
{
public:
	cudaGraphicsResource* cudaResource = nullptr;
	cudaSurfaceObject_t cudaSurface = 0;

	CudaGLBuffer() = default;

	void register_resource(GLuint image)
	{
		check(!cudaResource);
		CUDA_CHECKED_CALL cudaGraphicsGLRegisterImage(&cudaResource, image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		check(cudaResource);
	}
	void unregister_resource()
	{
		check(cudaResource);
		CUDA_CHECKED_CALL cudaGraphicsUnregisterResource(cudaResource);
		cudaResource = nullptr;
	}

	void map_surface()
	{
		check(!cudaSurface);

		CUDA_CHECKED_CALL cudaGraphicsMapResources(1, &cudaResource);
		cudaArray* cudaArray = nullptr;
		CUDA_CHECKED_CALL cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);
		create_surface(cudaArray);
		check(cudaSurface);
	}
	void unmap_surface()
	{
		check(cudaSurface);
		destroy_surface();
		CUDA_CHECKED_CALL cudaGraphicsUnmapResources(1, &cudaResource);
	}

	void create_surface(cudaArray* array)
    {
        check(!cudaSurface);
        check(array);
		cudaResourceDesc cudaArrayResourceDesc{};
        cudaArrayResourceDesc.resType = cudaResourceTypeArray;
        cudaArrayResourceDesc.res.array.array = array;
		CUDA_CHECKED_CALL cudaCreateSurfaceObject(&cudaSurface, &cudaArrayResourceDesc);
		check(cudaSurface);

    }
    void destroy_surface()
    {
        CUDA_CHECKED_CALL cudaDestroySurfaceObject(cudaSurface);
        cudaSurface = 0;
    }
};

