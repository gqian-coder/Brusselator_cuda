#include <mpi.h>
#include <vector>
#include <cstdio>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <adios2.h>
#include "mgard/compress_x.hpp"
#include "kernels.cuh"
#include "funInit.hpp"

static inline void ck(cudaError_t e){ if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} }

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int rank, nprocs; 
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  
  int cnt_argv = 1;
  
  std::string filename = argv[cnt_argv++];

  // initialization option: 
  // 0 -- restart
  // 1 -- shadow system
  // 2 -- 4 rain drops
  int init_fun = std::stoi(argv[cnt_argv++]);
  size_t init_ts   = 0;
  double zm_factor = 1.0;
  if (init_fun<2) {
    init_ts = std::stoi(argv[cnt_argv++]);
  } 
  if (init_fun==1) {
    // zm_factor > 1: load checkpoint and make it zm_factor times larger through repeat copy
    // zm_factor < 1: skip copy
    zm_factor = std::stof(argv[cnt_argv++]);
  }

  // --- domain (per-rank) ---
  double Lx = std::stof(argv[cnt_argv++]);
  double Ly = std::stof(argv[cnt_argv++]);
  double dh = std::stof(argv[cnt_argv++]);
  double dt = std::stof(argv[cnt_argv++]);
  double T  = std::stof(argv[cnt_argv++]);

  bool compression_cpt = std::stoi(argv[cnt_argv++]);
  double tol_u         = std::stof(argv[cnt_argv++]);
  double tol_v         = std::stof(argv[cnt_argv++]);
  double snorm         = 1;
  size_t wt_interval   = (size_t) std::stof(argv[cnt_argv++]);
   
  size_t steps = (size_t)(T/dt);
  size_t Nx = (size_t)std::ceil((double)Lx / dh) ;
  size_t Ny = (size_t)std::ceil((double)Ly / dh) ;

  const double inv_h2 = 1.0/(dh*dh);   // adjust if anisotropic
  
  // FitzHugh-Nagumo parameters
  const double Du = 1.0;      // Diffusion coefficient for u (voltage-like variable)
  const double Dv = 1.0;      // Diffusion coefficient for v (recovery variable)
  const double a = 0.1;       // Threshold parameter
  const double gamma = 1.0;   // Recovery parameter
  const double epsilon = 0.01; // Time scale separation (small value)

  // Check for CFL condition
  double dt_cfl = std::min(dh*dh/4.0/ Du, dh*dh/4.0/Dv);
  if (rank==0) std::cout << "CFL required dt " << dt_cfl << "\n";
  if (dt >= dt_cfl) {
      std::cout << "CFL condition violated\n";
      MPI_Finalize();
      return -1;
  }
  
  if (rank==0) {
    std::cout << "FitzHugh-Nagumo parameters:\n";
    std::cout << "Du = " << Du << ", Dv = " << Dv << "\n";
    std::cout << "a = " << a << ", gamma = " << gamma << ", epsilon = " << epsilon << "\n";
  }

  // --- select GPU ---
  int ngpu=0; ck(cudaGetDeviceCount(&ngpu));
  ck(cudaSetDevice(1));

  Grid2D g{Nx, Ny};

  // --- allocate device arrays ---
  size_t nTot = Nx * Ny;
  double *u_d, *v_d, *ut, *vt, *Lu_d, *Lv_d, *k1u,*k2u,*k3u,*k4u, *k1v,*k2v,*k3v,*k4v;
  size_t bytes = nTot * sizeof(double);
  std::cout << "number of steps: " << steps << ", domain size: " << Nx << " x " << Ny << "\n";
  ck(cudaMalloc(&u_d,  bytes)); ck(cudaMalloc(&v_d,  bytes));
  ck(cudaMalloc(&ut ,  bytes)); ck(cudaMalloc(&vt ,  bytes));
  ck(cudaMalloc(&Lu_d, bytes)); ck(cudaMalloc(&Lv_d, bytes));
  ck(cudaMalloc(&k1u, bytes));  ck(cudaMalloc(&k2u, bytes));
  ck(cudaMalloc(&k3u, bytes));  ck(cudaMalloc(&k4u, bytes));
  ck(cudaMalloc(&k1v, bytes));  ck(cudaMalloc(&k2v, bytes));
  ck(cudaMalloc(&k3v, bytes));  ck(cudaMalloc(&k4v, bytes));

  // Initialize ADIOS2
  adios2::ADIOS adios(MPI_COMM_WORLD);
  adios2::IO io = adios.DeclareIO("FitzHugh-Nagumo");
  std::vector<std::size_t> shape = {(std::size_t)Nx, (std::size_t)Ny};
  std::vector<std::size_t> start = {(std::size_t)(0), (std::size_t)(0)};
  std::vector<std::size_t> count = {(std::size_t)Nx, (std::size_t)Ny};

  // --- host init + H2D ---
  // Initialize with typical FitzHugh-Nagumo resting state
  std::vector<double> u_h(nTot, 0.0), v_h(nTot, 0.0); // Resting state
  if (init_fun<2) {
      // Get the original data size
      size_t zm_Nx = (std::size_t)(static_cast<double>(Nx) / zm_factor);
      size_t zm_Ny = (std::size_t)(static_cast<double>(Ny) / zm_factor);
      std::vector<double> in_var(zm_Nx * zm_Ny);
      std::vector<std::size_t> zm_shape = {zm_Nx, zm_Ny};
      std::vector<std::size_t> zm_start = {(std::size_t)(0), (std::size_t)(0)};
      std::vector<std::size_t> zm_count = {zm_Nx, zm_Ny};
      
      adios2::IO reader_io = adios.DeclareIO("Input");
      reader_io.SetEngine("BP");
      adios2::Engine reader = reader_io.Open(filename, adios2::Mode::ReadRandomAccess);
      adios2::Variable<double> variable_u, variable_v;
      variable_u = reader_io.InquireVariable<double>("u");
      variable_v = reader_io.InquireVariable<double>("v");
      std::cout << "Initialize the simulation from a previously saved checkpoint file...\n";
      std::cout << "total number of steps: " << variable_u.Steps() << ", read from " << init_ts << " timestep \n";
      // --- u ----
      variable_u.SetSelection({zm_start, zm_count});
      variable_u.SetStepSelection({init_ts, 1});
      reader.Get(variable_u, in_var.data());
      reader.PerformGets();
      if (init_fun == 0) {
          std::copy(in_var.data(), in_var.data() + nTot, u_h.data());
      } else { // shadow system w/ downsampled resolution
          dwSampleCopy(in_var.data(), u_h.data(), zm_factor, Nx, Ny);
      } 
      // --- v ----
      variable_v.SetSelection({zm_start, zm_count});
      variable_v.SetStepSelection({init_ts, 1});
      reader.Get(variable_v, in_var.data());
      reader.PerformGets();
      if (init_fun == 0) {
          std::copy(in_var.data(), in_var.data() + nTot, v_h.data());
      } else { // shadow system w/ downsampled resolution
          dwSampleCopy(in_var.data(), v_h.data(), zm_factor, Nx, Ny);
      }
     
      reader.Close();
      in_var.clear();

      filename.erase(filename.size() - 3);
      if (compression_cpt) {
        filename.append("-eb.bp");
      } else if (init_fun==0){
        filename.append("-cr.bp");
      } else {
        std::cout << " Create a zoomed in/out shadow system w/ a factor of " << zm_factor << "\n";
        filename.append("-shadow-cr.bp");
      }
  } else if (init_fun==2) {
      init_fields_rainDrops(u_h.data(), v_h.data(), Nx, Ny); // funInit.
  }
  ck(cudaMemcpy(u_d, u_h.data(), bytes, cudaMemcpyHostToDevice));
  ck(cudaMemcpy(v_d, v_h.data(), bytes, cudaMemcpyHostToDevice));

  auto var_u = io.DefineVariable<double>("u", shape, start, count);
  auto var_v = io.DefineVariable<double>("v", shape, start, count);

  dim3 bs(16,16);
  dim3 gs( (Nx+bs.x-1)/bs.x, (Ny+bs.y-1)/bs.y );

  void *compressed_u = NULL, *compressed_v = NULL;
  std::vector<mgard_x::SIZE> mgard_shape{Nx, Ny};
  size_t compressed_size_u = 0, compressed_size_v = 0, total_u_bytes = 0, temp_cp_size;
  // compression parameters
  mgard_x::Config config;
  if (compression_cpt) {
    compressed_u = (void *)malloc(bytes);
    compressed_v = (void *)malloc(bytes);
    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
    config.normalize_coordinates = true;
    config.dev_type = mgard_x::device_type::SERIAL;
  }

  std::cout << "start loop\n";
  adios2::Engine writer = io.Open(filename, adios2::Mode::Write);

  for(int t=0; t<steps; ++t){
    // write the timestep 0 before simulation update
    if (t % wt_interval == 0) {
      std::cout << "step " << t << "..." << static_cast<double>(t) / static_cast<double>(steps) * 100.0 << "% \n";
      ck(cudaMemcpy(u_h.data(), u_d, bytes, cudaMemcpyDeviceToHost));
      ck(cudaMemcpy(v_h.data(), v_d, bytes, cudaMemcpyDeviceToHost));
      writer.BeginStep();
      if (compression_cpt) {
        // ---- Compress u ----- 
        size_t temp_mgr_size = bytes;
        std::cout << "compress " << bytes << " bytes\n";
        mgard_x::compress(2, mgard_x::data_type::Double, mgard_shape, tol_u, snorm,
                mgard_x::error_bound_type::ABS, u_h.data(),
                compressed_u, temp_mgr_size, config, true);
        compressed_size_u += temp_mgr_size;
        // write out decompressed data to checkpoint file (so restart does not need to decompress)
        void *decompress_u = static_cast<void*>(u_h.data());
        mgard_x::decompress(compressed_u, temp_mgr_size, decompress_u, config, true);
        // ---- Compress v -----
        temp_mgr_size = bytes;
        mgard_x::compress(2, mgard_x::data_type::Double, mgard_shape, tol_v, snorm,
                mgard_x::error_bound_type::ABS, v_h.data(),
                compressed_v, temp_mgr_size, config, true);
        compressed_size_v += temp_mgr_size;
        void *decompress_v = static_cast<void*>(v_h.data());
        mgard_x::decompress(compressed_v, temp_mgr_size, decompress_v, config, true);
        total_u_bytes += bytes;
      }
      writer.Put(var_u, u_h.data(), adios2::Mode::Sync);
      writer.Put(var_v, v_h.data(), adios2::Mode::Sync);

      writer.PerformPuts();
      writer.EndStep();
    }
    
    // RK4 time integration using FitzHugh-Nagumo equations
    
    // ===== k1 ===== 
    laplacian2d<<<gs,bs>>>(u_d, Lu_d, g, inv_h2);
    laplacian2d<<<gs,bs>>>(v_d, Lv_d, g, inv_h2);
    fitzhugh_nagumo_rhs<<<gs,bs>>>(u_d, v_d, Lu_d, Lv_d, k1u, k1v, g, Du, Dv, a, gamma, epsilon);

    // ===== k2 ===== 
    // ut = u + (dt/2) * k1
    // vt = v + (dt/2) * k1
    axpy2d_full<<<gs, bs>>>(u_d, v_d, k1u, k1v, 0.5*dt, ut, vt, Nx, Ny);

    // Laplacians at tmp
    laplacian2d<<<gs,bs>>>(ut, Lu_d, g, inv_h2);
    laplacian2d<<<gs,bs>>>(vt, Lv_d, g, inv_h2);
    // k2 = f(ut, vt)
    fitzhugh_nagumo_rhs<<<gs,bs>>>(ut, vt, Lu_d, Lv_d, k2u, k2v, g, Du, Dv, a, gamma, epsilon);

    // ===== k3 =====
    // ut = u + (dt/2) * k2
    // vt = v + (dt/2) * k2
    axpy2d_full<<<gs, bs>>>(u_d, v_d, k2u, k2v, 0.5*dt, ut, vt, Nx, Ny);

    laplacian2d<<<gs,bs>>>(ut, Lu_d, g, inv_h2);
    laplacian2d<<<gs,bs>>>(vt, Lv_d, g, inv_h2);

    fitzhugh_nagumo_rhs<<<gs,bs>>>(ut, vt, Lu_d, Lv_d, k3u, k3v, g, Du, Dv, a, gamma, epsilon);

    // ===== k4 =====
    // ut = u + dt * k3
    // vt = v + dt * k3
    axpy2d_full<<<gs, bs>>>(u_d, v_d, k3u, k3v, dt, ut, vt, Nx, Ny);

    laplacian2d<<<gs,bs>>>(ut, Lu_d, g, inv_h2);
    laplacian2d<<<gs,bs>>>(vt, Lv_d, g, inv_h2);

    fitzhugh_nagumo_rhs<<<gs,bs>>>(ut, vt, Lu_d, Lv_d, k4u, k4v, g, Du, Dv, a, gamma, epsilon);

    rk4_combine<<<gs,bs>>>(u_d, v_d, k1u, k2u, k3u, k4u, k1v, k2v, k3v, k4v, g, dt);
  }

  writer.Close();
  if (compression_cpt) {
    free(compressed_u); 
    free(compressed_v);
  }

  cudaFree(k4v); cudaFree(k3v); cudaFree(k2v); cudaFree(k1v);
  cudaFree(k4u); cudaFree(k3u); cudaFree(k2u); cudaFree(k1u);
  cudaFree(Lv_d); cudaFree(Lu_d); cudaFree(v_d); cudaFree(u_d);
  cudaFree(vt); cudaFree(ut);

  MPI_Finalize();
  
  if (compression_cpt) {
    std::cout << "compression ratio CR(u) = " << static_cast<float>(total_u_bytes) / static_cast<float>(compressed_size_u) << ", CR(v) = " << static_cast<float>(total_u_bytes) / static_cast<float>(compressed_size_v) << "\n";
  }
  return 0;
}