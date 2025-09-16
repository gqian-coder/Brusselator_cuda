#include <random>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

// Helper to index 2D arrays stored in 1D
inline size_t idx_2d(size_t i, size_t j, size_t ny) { return i * ny + j; }

// only support zm_factor in the format of integer
// zm_factor: > 1: zm_var is the zoomed-in version of in_var; vice verse
// nx, ny: the dimension of zm_var
template <typename Real>
void dwSampleCopy(Real *in_var, Real *zm_var, Real zm_factor, size_t nx, size_t ny)
{
    size_t ny_in, offset_zm, offset_in;
    ny_in = static_cast<size_t>(static_cast<double>(ny) / zm_factor);
    for (size_t i=0; i<nx; i++) {
        size_t i_in = static_cast<size_t>(static_cast<double>(i) / zm_factor);
        offset_in   = i_in * ny_in;
        offset_zm   = i    * ny;
        for (size_t j=0; j<ny; j++) {
            size_t j_in = static_cast<size_t>(static_cast<double>(j) / zm_factor);
            zm_var[offset_zm + j] = in_var[offset_in + j_in];
        }
    }
}

template <typename Real>
void init_fields_rainDrops(Real * u, Real* v, size_t Nx, size_t Ny)
{
    size_t drop_width = static_cast<double>(Nx)/3.0;
    size_t n_drops    = 4;
    // Size of the Gaussian template each drop is based on.
    size_t cx = (size_t)(drop_width/2);
    size_t cy = (size_t)(drop_width/2);
    std::vector<Real> gauss_template(drop_width*drop_width, 0);
    Real max_intensity = 1.0;
    std::vector <double> px(drop_width), py(drop_width);
    for (size_t r=0; r<drop_width; r++) {
        px[r] = ((double)r - cx)/drop_width*16;
    }
    for (size_t c=0; c<drop_width; c++) {
        py[c] = (double(c)-cy)/drop_width*16;
    }
    size_t thresh = static_cast<size_t>(0.25*static_cast<double>(drop_width*drop_width));
    for (size_t r=0; r<drop_width; r++) {
        for (size_t c=0; c<drop_width; c++) {
            if ((r-cx)*(r-cx)+(c-cy)*(c-cy) < thresh) {
                gauss_template[r*drop_width+c] = max_intensity * exp(-(px[r]*px[r] + py[c]*py[c])/100.0);
            }
        }
    }
    fun_MultiRainDrop<Real>(u, v, Nx, Ny, drop_width, drop_width, gauss_template.data(), n_drops);
}


template <typename Real> 
Real src_Gaussian_pulse(Real freq, Real ts, Real A)  
{
    return (-2.0*A*ts * (freq*freq) * (std::exp(-(freq*freq) * (ts*ts)))) ;
}

// t0: nt * dt
template <typename Real>
void fun_Gaussian_pulse(Real *u, Real freq, Real t0, Real A, size_t xsrc, size_t ysrc, 
                    size_t zsrc, size_t Ny, size_t Nz)
{
    size_t k = xsrc * (Ny*Nz) + ysrc*Nz + zsrc;
    u[k] = src_Gaussian_pulse(freq, -t0, A); 
}


// an array of 2D plane wave starting from x=px 
template <typename Real>
void fun_cos_waves(Real *u, Real *v, size_t Nx, size_t Ny, Real A, Real freq)
{
    for (size_t i=1; i<Nx; i++) {
        for (size_t j=1; j<Ny; j++) {
            size_t id = idx_2d(i, j, Ny);
            Real x = i * freq;
            Real y = j  * freq;
            u[id] += A * cos(M_PI * x) * cos(M_PI * y); 
            v[id] += A * cos(2 * M_PI * x) * cos(2 * M_PI * y);
        }
    }
}

// drop_probability: Raindrop probability (with each time tick) and intensity.
// NDx, NDy: droplet's region
// one droplet at each time
// source global location 
template <typename Real>
void fun_rainDrop(Real *u, Real *v, size_t Nx, size_t Ny, size_t NDx, size_t NDy, 
                  Real *gauss_template)
{
    size_t cx = (size_t)(NDx/2);
    size_t cy = (size_t)(NDy/2);

    float random_x = 0.5;
    float random_y = 0.5;
    size_t x = static_cast<size_t>(random_x * Nx - cx);
    size_t y = static_cast<size_t>(random_y * Ny - cy);
    size_t dim1    = Ny; 
    size_t offset_x,  k; 
    size_t local_x, local_y;
    for (size_t r=0; r<NDx; r++) {
        local_x = r+x; 
        offset_x = local_x*dim1;
        for (size_t c=0; c<NDy; c++) {
            local_y = c+y;
            k = offset_x + local_y;
            u[k] += gauss_template[r*dim1+c];
            v[k] += gauss_template[r*dim1+c];
        }
    }
}

// drop_probability: Raindrop probability (with each time tick) and intensity.
// drop multiple droplets
template <typename Real>
void fun_MultiRainDrop(Real *u, Real *v, size_t Nx, size_t Ny, size_t NDx, size_t NDy, 
                       Real *gauss_template, size_t nDrops)
{
    size_t cx = (size_t)(NDx/2);
    size_t cy = (size_t)(NDy/2);
    // try not to generate rain drops closing to edges
    for (size_t d=0; d<nDrops; d++) {
        std::vector<float> px  = {0.275, 0.65, 0.35, 0.7};
        std::vector<float> py  = {0.35, 0.3, 0.725, 0.65};
        std::vector<float> mag = {0.7, 0.5, 0.85, 0.6};
        //std::vector<float> px  = {0.375, 0.65, 0.45, 0.63};
        //std::vector<float> py  = {0.45, 0.4, 0.725, 0.65};
        //std::vector<float> pz  = {0.44, 0.56, 0.75, 0.35};
        //std::vector<float> mag = {0.6, 0.5, 0.65, 0.6};
        float random_x = px[d];
        float random_y = py[d];
        size_t x = static_cast<size_t>(random_x * Nx - cx);
        size_t y = static_cast<size_t>(random_y * Ny - cy);
        std::cout << "x, y = " << x << ", " << y  << ", " << cx << ", " << cy << ", "  << NDx << ", " << NDy << ", " << Nx << ", " << Ny << "\n";
        size_t dim1  = Ny; 
        size_t dimD1 = NDy;
        size_t offset_x, k;
        size_t local_x, local_y;
        float intensity = mag[d];// dis(gen);
        for (size_t r=0; r<NDx; r++) {
            local_x = r+x;
            offset_x = local_x*dim1;
            for (size_t c=0; c<NDy; c++) {
                local_y  = c+y;
                k = offset_x + local_y; 
                u[k] += intensity*gauss_template[r*dimD1+c];
                v[k] += intensity*gauss_template[r*dimD1+c];
            }
        }
    }
}

