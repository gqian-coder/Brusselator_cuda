#pragma once
#include <cuda_runtime.h>

struct Grid2D {
  int nx, ny;      // interior
};


__device__ __forceinline__
int idx2D(int i, int j, const Grid2D& g) { 
    return i * g.ny + j; 
}

// 5-point Laplacian with periodic boundary
template <typename T>
__global__ void laplacian2d(const T* __restrict__ u, T* __restrict__ Lu,
                            Grid2D g, T inv_h2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x ; 
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  if (i >= g.nx || j >= g.ny) return;
  int id = idx2D(i, j, g);
  int ip = (i<g.nx-1) ? idx2D(i+1, j, g) : idx2D(0, j, g);
  int im = (i>0) ? idx2D(i-1, j, g) : idx2D(g.nx-1, j, g);
  int jp = (j<g.ny-1) ? idx2D(i, j+1, g) : idx2D(i, 0, g); 
  int jm = (j>0) ? idx2D(i, j-1, g) : idx2D(i, g.ny-1, g);
  Lu[id] = (u[im] + u[ip] + u[jm] + u[jp] - T(4)*u[id]) * inv_h2;
}

// Brusselator reaction RHS at each point
template <typename T>
__global__ void brusselator_rhs(const T* __restrict__ u,
                                const T* __restrict__ v,
                                const T* __restrict__ Lu,
                                const T* __restrict__ Lv,
                                T* __restrict__ fu,
                                T* __restrict__ fv,
                                Grid2D g, T Du, T Dv, T A, T B)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  if (i >= g.nx || j >= g.ny) return;
  int id = idx2D(i, j, g);
  T ui = u[id], vi = v[id];
  // du/dt = Du ∇²u + A - (B+1)u + u^2 v
  // dv/dt = Dv ∇²v + Bu - u^2 v
  fu[id] = Du * Lu[id] + (A - (B + T(1)) * ui + ui*ui*vi);
  fv[id] = Dv * Lv[id] + (B * ui - ui*ui*vi);
}

// Classic RK4 update in-place (stage combination)
// u_{n+1} = u_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
// Same for v
template <typename T>
__global__ void rk4_combine(T* __restrict__ u, T* __restrict__ v,
                            const T* __restrict__ k1u, const T* __restrict__ k2u,
                            const T* __restrict__ k3u, const T* __restrict__ k4u,
                            const T* __restrict__ k1v, const T* __restrict__ k2v,
                            const T* __restrict__ k3v, const T* __restrict__ k4v,
                            Grid2D g, T dt)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  if (i >= g.nx || j >= g.ny) return;
  int id = idx2D(i, j, g);
  T sixth = dt / T(6);
  u[id] += sixth * (k1u[id] + T(2)*(k2u[id] + k3u[id]) + k4u[id]);
  v[id] += sixth * (k1v[id] + T(2)*(k2v[id] + k3v[id]) + k4v[id]);
}

// y = x + a * k   (operate over the full buffer including ghosts)
template <typename T>
__global__ void axpy2d_full(const T* __restrict__ u,
                            const T* __restrict__ v,
                            const T* __restrict__ ku,
                            const T* __restrict__ kv,    
                            T a,
                            T* __restrict__ ut,
                            T* __restrict__ vt,
                            int pitch,   // leading dimension = nx 
                            int height)  // total rows = ny 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  int j = blockIdx.y * blockDim.y + threadIdx.y; 
  if (i >= pitch || j >= height) return;

  int id = i *  height + j; 
  ut[id] = u[id] + a * ku[id];
  vt[id] = v[id] + a * kv[id];
}

