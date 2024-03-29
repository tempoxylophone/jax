/* Copyright 2021 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file is not used by JAX itself, but exists to assist with running
// JAX-generated HLO code from outside of JAX.
#include "jaxlib/cpu/lapack_kernels.h"

namespace jax {
namespace {

namespace ffi = ::xla::ffi;

#define JAX_CPU_REGISTER_TRSM(name, data_type)                             \
  XLA_FFI_DEFINE_HANDLER(name, TriMatrixEquationSolver<data_type>::Kernel, \
                         ffi::Ffi::Bind()                                  \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)           \
                             .Arg<ffi::Buffer<data_type>>(/*y*/)           \
                             .Arg<ffi::Buffer<data_type>>(/*alpha*/)       \
                             .Ret<ffi::Buffer<data_type>>(/*y_out*/)       \
                             .Attr<MatrixParams::Side>("side")             \
                             .Attr<MatrixParams::UpLo>("uplo")             \
                             .Attr<MatrixParams::Transpose>("trans_x")     \
                             .Attr<MatrixParams::Diag>("diag"));           \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_GETRF(name, data_type)                            \
  XLA_FFI_DEFINE_HANDLER(name, LuDecomposition<data_type>::Kernel,         \
                         ffi::Ffi::Bind()                                  \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)           \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)       \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*ipiv*/)   \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/)); \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_GEQRF(name, data_type)                          \
  XLA_FFI_DEFINE_HANDLER(name, QrFactorization<data_type>::Kernel,       \
                         ffi::Ffi::Bind()                                \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)         \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)     \
                             .Ret<ffi::Buffer<data_type>>(/*tau*/)       \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/) \
                             .Ret<ffi::Buffer<data_type>>(/*work*/));    \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_ORGQR(name, data_type)                          \
  XLA_FFI_DEFINE_HANDLER(name, OrthogonalQr<data_type>::Kernel,          \
                         ffi::Ffi::Bind()                                \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)         \
                             .Arg<ffi::Buffer<data_type>>(/*tau*/)       \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)     \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/) \
                             .Ret<ffi::Buffer<data_type>>(/*work*/));    \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_POTRF(name, data_type)                            \
  XLA_FFI_DEFINE_HANDLER(name, CholeskyFactorization<data_type>::Kernel,   \
                         ffi::Ffi::Bind()                                  \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)           \
                             .Attr<MatrixParams::UpLo>("uplo")             \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)       \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/)); \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_GESDD(name, data_type)                               \
  XLA_FFI_DEFINE_HANDLER(name, SingularValueDecomposition<data_type>::Kernel, \
                         ffi::Ffi::Bind()                                     \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)              \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)          \
                             .Ret<ffi::Buffer<data_type>>(/*s*/)              \
                             .Ret<ffi::Buffer<data_type>>(/*u*/)              \
                             .Ret<ffi::Buffer<data_type>>(/*vt*/)             \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/)      \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*iwork*/)     \
                             .Ret<ffi::Buffer<data_type>>(/*work*/)           \
                             .Attr<svd::ComputationMode>("mode"));            \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_GESDD_COMPLEX(name, data_type)           \
  XLA_FFI_DEFINE_HANDLER(                                         \
      name, SingularValueDecompositionComplex<data_type>::Kernel, \
      ffi::Ffi::Bind()                                            \
          .Arg<ffi::Buffer<data_type>>(/*x*/)                     \
          .Ret<ffi::Buffer<data_type>>(/*x_out*/)                 \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*s*/)        \
          .Ret<ffi::Buffer<data_type>>(/*u*/)                     \
          .Ret<ffi::Buffer<data_type>>(/*vt*/)                    \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/)             \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*rwork*/)    \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*iwork*/)            \
          .Ret<ffi::Buffer<data_type>>(/*work*/)                  \
          .Attr<svd::ComputationMode>("mode"));                   \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_SYEVD(name, data_type)                               \
  XLA_FFI_DEFINE_HANDLER(name,                                                \
                         EigenvalueDecompositionSymmetric<data_type>::Kernel, \
                         ffi::Ffi::Bind()                                     \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)              \
                             .Attr<MatrixParams::UpLo>("uplo")                \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)          \
                             .Ret<ffi::Buffer<data_type>>(/*eigenvalues*/)    \
                             .Ret<ffi::Buffer<data_type>>(/*work*/)           \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*iwork*/)     \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/)      \
                             .Attr<eig::ComputationMode>("mode"));            \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_HEEVD(name, data_type)                      \
  XLA_FFI_DEFINE_HANDLER(                                            \
      name, EigenvalueDecompositionHermitian<data_type>::Kernel,     \
      ffi::Ffi::Bind()                                               \
          .Arg<ffi::Buffer<data_type>>(/*x*/)                        \
          .Attr<MatrixParams::UpLo>("uplo")                          \
          .Ret<ffi::Buffer<data_type>>(/*x_out*/)                    \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*eigenvalues*/) \
          .Ret<ffi::Buffer<data_type>>(/*work*/)                     \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*rwork*/)       \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*iwork*/)               \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/)                \
          .Attr<eig::ComputationMode>("mode"));                      \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_GEEV(name, data_type)                                \
  XLA_FFI_DEFINE_HANDLER(                                                     \
      name, EigenvalueDecomposition<data_type>::Kernel,                       \
      ffi::Ffi::Bind()                                                        \
          .Arg<ffi::Buffer<data_type>>(/*x*/)                                 \
          .Attr<eig::ComputationMode>("compute_left")                         \
          .Attr<eig::ComputationMode>("compute_right")                        \
          .Ret<ffi::Buffer<data_type>>(/*x_out*/)                             \
          .Ret<ffi::Buffer<data_type>>(/*eigvals_real*/)                      \
          .Ret<ffi::Buffer<data_type>>(/*eigvals_imag*/)                      \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*eigvecs_left*/)         \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*eigvecs_right*/)        \
          .Ret<ffi::Buffer<ffi::ToComplex(data_type)>>(/*eigvecs_left_out*/)  \
          .Ret<ffi::Buffer<ffi::ToComplex(data_type)>>(/*eigvecs_right_out*/) \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/));                       \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_GEEV_COMPLEX(name, data_type)         \
  XLA_FFI_DEFINE_HANDLER(                                      \
      name, EigenvalueDecompositionComplex<data_type>::Kernel, \
      ffi::Ffi::Bind()                                         \
          .Arg<ffi::Buffer<data_type>>(/*x*/)                  \
          .Attr<eig::ComputationMode>("compute_left")          \
          .Attr<eig::ComputationMode>("compute_right")         \
          .Ret<ffi::Buffer<data_type>>(/*x_out*/)              \
          .Ret<ffi::Buffer<data_type>>(/*eigvals*/)            \
          .Ret<ffi::Buffer<data_type>>(/*eigvecs_left*/)       \
          .Ret<ffi::Buffer<data_type>>(/*eigvecs_right*/)      \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*rwork*/) \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/));        \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_GEES(name, data_type)                        \
  XLA_FFI_DEFINE_HANDLER(                                             \
      name, SchurDecomposition<data_type>::Kernel,                    \
      ffi::Ffi::Bind()                                                \
          .Arg<ffi::Buffer<data_type>>(/*x*/)                         \
          .Attr<schur::ComputationMode>("mode")                       \
          .Attr<schur::Sort>("sort")                                  \
          .Ret<ffi::Buffer<data_type>>(/*x_out*/)                     \
          .Ret<ffi::Buffer<data_type>>(/*eigvals_real*/)              \
          .Ret<ffi::Buffer<data_type>>(/*eigvals_imag*/)              \
          .Ret<ffi::Buffer<data_type>>(/*schur_vectors*/)             \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*selected_eigval_dims*/) \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/));               \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_GEES_COMPLEX(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER(                                             \
      name, SchurDecompositionComplex<data_type>::Kernel,             \
      ffi::Ffi::Bind()                                                \
          .Arg<ffi::Buffer<data_type>>(/*x*/)                         \
          .Attr<schur::ComputationMode>("mode")                       \
          .Attr<schur::Sort>("sort")                                  \
          .Ret<ffi::Buffer<data_type>>(/*x_out*/)                     \
          .Ret<ffi::Buffer<data_type>>(/*eigvals*/)                   \
          .Ret<ffi::Buffer<data_type>>(/*schur_vectors*/)             \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*selected_eigval_dims*/) \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*rwork*/)        \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/));               \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_GEHRD(name, data_type)                            \
  XLA_FFI_DEFINE_HANDLER(name, HessenbergDecomposition<data_type>::Kernel, \
                         ffi::Ffi::Bind()                                  \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)           \
                             .Attr<lapack_int>("low")                      \
                             .Attr<lapack_int>("high")                     \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)       \
                             .Ret<ffi::Buffer<data_type>>(/*tau*/)         \
                             .Ret<ffi::Buffer<data_type>>(/*work*/)        \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/)); \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

#define JAX_CPU_REGISTER_SYTRD_HETRD(name, data_type)                 \
  XLA_FFI_DEFINE_HANDLER(                                             \
      name, TridiagonalReduction<data_type>::Kernel,                  \
      ffi::Ffi::Bind()                                                \
          .Arg<ffi::Buffer<data_type>>(/*x*/)                         \
          .Attr<MatrixParams::UpLo>("uplo")                           \
          .Ret<ffi::Buffer<data_type>>(/*x_out*/)                     \
          .Ret<ffi::Buffer<data_type>>(/*tau*/)                       \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*diagonal*/)     \
          .Ret<ffi::Buffer<ffi::ToReal(data_type)>>(/*off_diagonal*/) \
          .Ret<ffi::Buffer<data_type>>(/*work*/)                      \
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/));               \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

JAX_CPU_REGISTER_TRSM(blas_strsm, ffi::DataType::F32);
JAX_CPU_REGISTER_TRSM(blas_dtrsm, ffi::DataType::F64);
JAX_CPU_REGISTER_TRSM(blas_ctrsm, ffi::DataType::C64);
JAX_CPU_REGISTER_TRSM(blas_ztrsm, ffi::DataType::C128);

JAX_CPU_REGISTER_GETRF(lapack_sgetrf, ffi::DataType::F32);
JAX_CPU_REGISTER_GETRF(lapack_dgetrf, ffi::DataType::F64);
JAX_CPU_REGISTER_GETRF(lapack_cgetrf, ffi::DataType::C64);
JAX_CPU_REGISTER_GETRF(lapack_zgetrf, ffi::DataType::C128);

JAX_CPU_REGISTER_GEQRF(lapack_sgeqrf, ffi::DataType::F32);
JAX_CPU_REGISTER_GEQRF(lapack_dgeqrf, ffi::DataType::F64);
JAX_CPU_REGISTER_GEQRF(lapack_cgeqrf, ffi::DataType::C64);
JAX_CPU_REGISTER_GEQRF(lapack_zgeqrf, ffi::DataType::C128);

JAX_CPU_REGISTER_ORGQR(lapack_sorgqr, ffi::DataType::F32);
JAX_CPU_REGISTER_ORGQR(lapack_dorgqr, ffi::DataType::F64);
JAX_CPU_REGISTER_ORGQR(lapack_cungqr, ffi::DataType::C64);
JAX_CPU_REGISTER_ORGQR(lapack_zungqr, ffi::DataType::C128);

JAX_CPU_REGISTER_POTRF(lapack_spotrf, ffi::DataType::F32);
JAX_CPU_REGISTER_POTRF(lapack_dpotrf, ffi::DataType::F64);
JAX_CPU_REGISTER_POTRF(lapack_cpotrf, ffi::DataType::C64);
JAX_CPU_REGISTER_POTRF(lapack_zpotrf, ffi::DataType::C128);

JAX_CPU_REGISTER_GESDD(lapack_sgesdd, ffi::DataType::F32);
JAX_CPU_REGISTER_GESDD(lapack_dgesdd, ffi::DataType::F64);
JAX_CPU_REGISTER_GESDD_COMPLEX(lapack_cgesdd, ffi::DataType::C64);
JAX_CPU_REGISTER_GESDD_COMPLEX(lapack_zgesdd, ffi::DataType::C128);

JAX_CPU_REGISTER_SYEVD(lapack_ssyevd, ffi::DataType::F32);
JAX_CPU_REGISTER_SYEVD(lapack_dsyevd, ffi::DataType::F64);
JAX_CPU_REGISTER_HEEVD(lapack_cheevd, ffi::DataType::C64);
JAX_CPU_REGISTER_HEEVD(lapack_zheevd, ffi::DataType::C128);

JAX_CPU_REGISTER_GEEV(lapack_sgeev, ffi::DataType::F32);
JAX_CPU_REGISTER_GEEV(lapack_dgeev, ffi::DataType::F64);
JAX_CPU_REGISTER_GEEV_COMPLEX(lapack_cgeev, ffi::DataType::C64);
JAX_CPU_REGISTER_GEEV_COMPLEX(lapack_zgeev, ffi::DataType::C128);

JAX_CPU_REGISTER_GEES(lapack_sgees, ffi::DataType::F32);
JAX_CPU_REGISTER_GEES(lapack_dgees, ffi::DataType::F64);
JAX_CPU_REGISTER_GEES_COMPLEX(lapack_cgees, ffi::DataType::C64);
JAX_CPU_REGISTER_GEES_COMPLEX(lapack_zgees, ffi::DataType::C128);

JAX_CPU_REGISTER_GEHRD(lapack_sgehrd, ffi::DataType::F32);
JAX_CPU_REGISTER_GEHRD(lapack_dgehrd, ffi::DataType::F64);
JAX_CPU_REGISTER_GEHRD(lapack_cgehrd, ffi::DataType::C64);
JAX_CPU_REGISTER_GEHRD(lapack_zgehrd, ffi::DataType::C128);

JAX_CPU_REGISTER_SYTRD_HETRD(lapack_ssytrd, ffi::DataType::F32);
JAX_CPU_REGISTER_SYTRD_HETRD(lapack_dsytrd, ffi::DataType::F64);
JAX_CPU_REGISTER_SYTRD_HETRD(lapack_chetrd, ffi::DataType::C64);
JAX_CPU_REGISTER_SYTRD_HETRD(lapack_zhetrd, ffi::DataType::C128);

#undef JAX_CPU_REGISTER_TRSM
#undef JAX_CPU_REGISTER_GETRF
#undef JAX_CPU_REGISTER_GEQRF
#undef JAX_CPU_REGISTER_ORGQR
#undef JAX_CPU_REGISTER_POTRF
#undef JAX_CPU_REGISTER_GESDD
#undef JAX_CPU_REGISTER_GESDD_COMPLEX
#undef JAX_CPU_REGISTER_SYEVD
#undef JAX_CPU_REGISTER_HEEVD
#undef JAX_CPU_REGISTER_GEEV
#undef JAX_CPU_REGISTER_GEEV_COMPLEX
#undef JAX_CPU_REGISTER_GEES
#undef JAX_CPU_REGISTER_GEES_COMPLEX
#undef JAX_CPU_REGISTER_GEHRD
#undef JAX_CPU_REGISTER_SYTRD_HETRD

}  // namespace
}  // namespace jax
