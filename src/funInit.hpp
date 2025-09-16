#ifndef FUNINIT_HPP
#define FUNINIT_HPP

template <typename Real>
Real src_Gaussian_pulse(Real freq, Real ts, Real A);

template <typename Real>
void fun_Gaussian_pulse(Real *u, Real freq, Real t0, Real A, size_t xsrc, size_t ysrc,
                    size_t zsrc, size_t Ny, size_t Nz);

template <typename Real>
void fun_cos_waves(Real *u, Real *v, size_t Nx, size_t Ny, Real A, Real freq);

template <typename Real>
void fun_rainDrop(Real *u, Real *v, size_t Nx, size_t Ny, 
                  size_t NDx, size_t NDy, Real *gauss_template);

template <typename Real>
void fun_MultiRainDrop(Real *u, Real *v, size_t Nx, size_t Ny, size_t NDx, size_t NDy,
                       Real *gauss_template, size_t nDrops);

template <typename Real>
void dwSampleCopy(Real *in_var, Real *zm_var, Real zm_factor, size_t nx, size_t ny);

template <typename Real>
void init_fields_rainDrops(Real * u, Real* v, size_t Nx, size_t Ny);

#include "funInit.tpp"
#endif
