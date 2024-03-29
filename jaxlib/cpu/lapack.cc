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

#include "nanobind/nanobind.h"
#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace {

namespace nb = nanobind;
namespace ffi = ::xla::ffi;

using ::xla::ffi::DataType;

// FFI Definition Macros (by DataType)

#define JAX_CPU_DEFINE_TRSM(name, data_type)                               \
  XLA_FFI_DEFINE_HANDLER(name, TriMatrixEquationSolver<data_type>::Kernel, \
                         ffi::Ffi::Bind()                                  \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)           \
                             .Arg<ffi::Buffer<data_type>>(/*y*/)           \
                             .Arg<ffi::Buffer<data_type>>(/*alpha*/)       \
                             .Ret<ffi::Buffer<data_type>>(/*y_out*/)       \
                             .Attr<MatrixParams::Side>("side")             \
                             .Attr<MatrixParams::UpLo>("uplo")             \
                             .Attr<MatrixParams::Transpose>("trans_x")     \
                             .Attr<MatrixParams::Diag>("diag"));

#define JAX_CPU_DEFINE_GETRF(name, data_type)                            \
  XLA_FFI_DEFINE_HANDLER(name, LuDecomposition<data_type>::Kernel,       \
                         ffi::Ffi::Bind()                                \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)         \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)     \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*ipiv*/) \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEQRF(name, data_type)                            \
  XLA_FFI_DEFINE_HANDLER(name, QrFactorization<data_type>::Kernel,       \
                         ffi::Ffi::Bind()                                \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)         \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)     \
                             .Ret<ffi::Buffer<data_type>>(/*tau*/)       \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/) \
                             .Ret<ffi::Buffer<data_type>>(/*work*/))

#define JAX_CPU_DEFINE_ORGQR(name, data_type)                            \
  XLA_FFI_DEFINE_HANDLER(name, OrthogonalQr<data_type>::Kernel,          \
                         ffi::Ffi::Bind()                                \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)         \
                             .Arg<ffi::Buffer<data_type>>(/*tau*/)       \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)     \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/) \
                             .Ret<ffi::Buffer<data_type>>(/*work*/))

#define JAX_CPU_DEFINE_POTRF(name, data_type)                            \
  XLA_FFI_DEFINE_HANDLER(name, CholeskyFactorization<data_type>::Kernel, \
                         ffi::Ffi::Bind()                                \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)         \
                             .Attr<MatrixParams::UpLo>("uplo")           \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)     \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GESDD(name, data_type)                                 \
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
                             .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GESDD_COMPLEX(name, data_type)             \
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
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_SYEVD(name, data_type)                                 \
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
                             .Attr<eig::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_HEEVD(name, data_type)                        \
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
          .Attr<eig::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GEEV(name, data_type)                                  \
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
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEEV_COMPLEX(name, data_type)           \
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
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEES(name, data_type)                          \
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
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEES_COMPLEX(name, data_type)                  \
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
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEHRD(name, data_type)                              \
  XLA_FFI_DEFINE_HANDLER(name, HessenbergDecomposition<data_type>::Kernel, \
                         ffi::Ffi::Bind()                                  \
                             .Arg<ffi::Buffer<data_type>>(/*x*/)           \
                             .Attr<lapack_int>("low")                      \
                             .Attr<lapack_int>("high")                     \
                             .Ret<ffi::Buffer<data_type>>(/*x_out*/)       \
                             .Ret<ffi::Buffer<data_type>>(/*tau*/)         \
                             .Ret<ffi::Buffer<data_type>>(/*work*/)        \
                             .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_SYTRD_HETRD(name, data_type)                   \
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
          .Ret<ffi::Buffer<LapackIntDtype>>(/*info*/))

// FFI Handlers

JAX_CPU_DEFINE_TRSM(blas_strsm, DataType::F32);
JAX_CPU_DEFINE_TRSM(blas_dtrsm, DataType::F64);
JAX_CPU_DEFINE_TRSM(blas_ctrsm, DataType::C64);
JAX_CPU_DEFINE_TRSM(blas_ztrsm, DataType::C128);

JAX_CPU_DEFINE_GETRF(lapack_sgetrf, DataType::F32);
JAX_CPU_DEFINE_GETRF(lapack_dgetrf, DataType::F64);
JAX_CPU_DEFINE_GETRF(lapack_cgetrf, DataType::C64);
JAX_CPU_DEFINE_GETRF(lapack_zgetrf, DataType::C128);

JAX_CPU_DEFINE_GEQRF(lapack_sgeqrf, DataType::F32);
JAX_CPU_DEFINE_GEQRF(lapack_dgeqrf, DataType::F64);
JAX_CPU_DEFINE_GEQRF(lapack_cgeqrf, DataType::C64);
JAX_CPU_DEFINE_GEQRF(lapack_zgeqrf, DataType::C128);

JAX_CPU_DEFINE_ORGQR(lapack_sorgqr, DataType::F32);
JAX_CPU_DEFINE_ORGQR(lapack_dorgqr, DataType::F64);
JAX_CPU_DEFINE_ORGQR(lapack_cungqr, DataType::C64);
JAX_CPU_DEFINE_ORGQR(lapack_zungqr, DataType::C128);

JAX_CPU_DEFINE_POTRF(lapack_spotrf, DataType::F32);
JAX_CPU_DEFINE_POTRF(lapack_dpotrf, DataType::F64);
JAX_CPU_DEFINE_POTRF(lapack_cpotrf, DataType::C64);
JAX_CPU_DEFINE_POTRF(lapack_zpotrf, DataType::C128);

JAX_CPU_DEFINE_GESDD(lapack_sgesdd, DataType::F32);
JAX_CPU_DEFINE_GESDD(lapack_dgesdd, DataType::F64);
JAX_CPU_DEFINE_GESDD_COMPLEX(lapack_cgesdd, DataType::C64);
JAX_CPU_DEFINE_GESDD_COMPLEX(lapack_zgesdd, DataType::C128);

JAX_CPU_DEFINE_SYEVD(lapack_ssyevd, DataType::F32);
JAX_CPU_DEFINE_SYEVD(lapack_dsyevd, DataType::F64);
JAX_CPU_DEFINE_HEEVD(lapack_cheevd, DataType::C64);
JAX_CPU_DEFINE_HEEVD(lapack_zheevd, DataType::C128);

JAX_CPU_DEFINE_GEEV(lapack_sgeev, DataType::F32);
JAX_CPU_DEFINE_GEEV(lapack_dgeev, DataType::F64);
JAX_CPU_DEFINE_GEEV_COMPLEX(lapack_cgeev, DataType::C64);
JAX_CPU_DEFINE_GEEV_COMPLEX(lapack_zgeev, DataType::C128);

JAX_CPU_DEFINE_GEES(lapack_sgees, DataType::F32);
JAX_CPU_DEFINE_GEES(lapack_dgees, DataType::F64);
JAX_CPU_DEFINE_GEES_COMPLEX(lapack_cgees, DataType::C64);
JAX_CPU_DEFINE_GEES_COMPLEX(lapack_zgees, DataType::C128);

JAX_CPU_DEFINE_GEHRD(lapack_sgehrd, DataType::F32);
JAX_CPU_DEFINE_GEHRD(lapack_dgehrd, DataType::F64);
JAX_CPU_DEFINE_GEHRD(lapack_cgehrd, DataType::C64);
JAX_CPU_DEFINE_GEHRD(lapack_zgehrd, DataType::C128);

JAX_CPU_DEFINE_SYTRD_HETRD(lapack_ssytrd, DataType::F32);
JAX_CPU_DEFINE_SYTRD_HETRD(lapack_dsytrd, DataType::F64);
JAX_CPU_DEFINE_SYTRD_HETRD(lapack_chetrd, DataType::C64);
JAX_CPU_DEFINE_SYTRD_HETRD(lapack_zhetrd, DataType::C128);

#undef JAX_CPU_DEFINE_TRSM
#undef JAX_CPU_DEFINE_GETRF
#undef JAX_CPU_DEFINE_GEQRF
#undef JAX_CPU_DEFINE_ORGQR
#undef JAX_CPU_DEFINE_POTRF
#undef JAX_CPU_DEFINE_GESDD
#undef JAX_CPU_DEFINE_GESDD_COMPLEX
#undef JAX_CPU_DEFINE_SYEVD
#undef JAX_CPU_DEFINE_HEEVD
#undef JAX_CPU_DEFINE_GEEV
#undef JAX_CPU_DEFINE_GEEV_COMPLEX
#undef JAX_CPU_DEFINE_GEES
#undef JAX_CPU_DEFINE_GEES_COMPLEX
#undef JAX_CPU_DEFINE_GEHRD
#undef JAX_CPU_DEFINE_SYTRD_HETRD

svd::ComputationMode GetSvdComputationMode(bool job_opt_compute_uv,
                                           bool job_opt_full_matrices) {
  if (!job_opt_compute_uv) {
    return svd::ComputationMode::kNoComputeUVt;
  } else if (!job_opt_full_matrices) {
    return svd::ComputationMode::kComputeMinUVt;
  }
  return svd::ComputationMode::kComputeFullUVt;
}

template <DataType dtype>
int64_t GesddGetWorkspaceSize(lapack_int m, lapack_int n,
                              bool job_opt_compute_uv,
                              bool job_opt_full_matrices) {
  svd::ComputationMode mode =
      GetSvdComputationMode(job_opt_compute_uv, job_opt_full_matrices);
  return svd::SVDType<dtype>::GetWorkspaceSize(m, n, mode);
};

lapack_int GesddGetRealWorkspaceSize(lapack_int m, lapack_int n,
                                     bool job_opt_compute_uv) {
  svd::ComputationMode mode = GetSvdComputationMode(job_opt_compute_uv, true);
  return svd::GetRealWorkspaceSize(m, n, mode);
}

// TODO(paruzelp): For some reason JAX prefers to assume a larger workspace
//                 Might need to investigate if that is necessary.
template <lapack_int (&f)(int64_t, eig::ComputationMode)>
inline constexpr auto BoundWithEigvecs = +[](lapack_int n) {
  return f(n, eig::ComputationMode::kComputeEigenvectors);
};

void GetLapackKernelsFromScipy() {
  static bool initialized = false;  // Protected by GIL
  if (initialized) return;
  nb::module_ cython_blas = nb::module_::import_("scipy.linalg.cython_blas");
  // Technically this is a Cython-internal API. However, it seems highly likely
  // it will remain stable because Cython itself needs API stability for
  // cross-package imports to work in the first place.
  nb::dict blas_capi = cython_blas.attr("__pyx_capi__");
  auto blas_ptr = [&](const char* name) {
    return nb::cast<nb::capsule>(blas_capi[name]).data();
  };

  AssignKernelFn<TriMatrixEquationSolver<DataType::F32>>(blas_ptr("strsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::F64>>(blas_ptr("dtrsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::C64>>(blas_ptr("ctrsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::C128>>(blas_ptr("ztrsm"));

  nb::module_ cython_lapack =
      nb::module_::import_("scipy.linalg.cython_lapack");
  nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");
  auto lapack_ptr = [&](const char* name) {
    return nb::cast<nb::capsule>(lapack_capi[name]).data();
  };
  AssignKernelFn<LuDecomposition<DataType::F32>>(lapack_ptr("sgetrf"));
  AssignKernelFn<LuDecomposition<DataType::F64>>(lapack_ptr("dgetrf"));
  AssignKernelFn<LuDecomposition<DataType::C64>>(lapack_ptr("cgetrf"));
  AssignKernelFn<LuDecomposition<DataType::C128>>(lapack_ptr("zgetrf"));

  AssignKernelFn<QrFactorization<DataType::F32>>(lapack_ptr("sgeqrf"));
  AssignKernelFn<QrFactorization<DataType::F64>>(lapack_ptr("dgeqrf"));
  AssignKernelFn<QrFactorization<DataType::C64>>(lapack_ptr("cgeqrf"));
  AssignKernelFn<QrFactorization<DataType::C128>>(lapack_ptr("zgeqrf"));

  AssignKernelFn<OrthogonalQr<DataType::F32>>(lapack_ptr("sorgqr"));
  AssignKernelFn<OrthogonalQr<DataType::F64>>(lapack_ptr("dorgqr"));
  AssignKernelFn<OrthogonalQr<DataType::C64>>(lapack_ptr("cungqr"));
  AssignKernelFn<OrthogonalQr<DataType::C128>>(lapack_ptr("zungqr"));

  AssignKernelFn<CholeskyFactorization<DataType::F32>>(lapack_ptr("spotrf"));
  AssignKernelFn<CholeskyFactorization<DataType::F64>>(lapack_ptr("dpotrf"));
  AssignKernelFn<CholeskyFactorization<DataType::C64>>(lapack_ptr("cpotrf"));
  AssignKernelFn<CholeskyFactorization<DataType::C128>>(lapack_ptr("zpotrf"));

  AssignKernelFn<SingularValueDecomposition<DataType::F32>>(
      lapack_ptr("sgesdd"));
  AssignKernelFn<SingularValueDecomposition<DataType::F64>>(
      lapack_ptr("dgesdd"));
  AssignKernelFn<SingularValueDecompositionComplex<DataType::C64>>(
      lapack_ptr("cgesdd"));
  AssignKernelFn<SingularValueDecompositionComplex<DataType::C128>>(
      lapack_ptr("zgesdd"));

  AssignKernelFn<EigenvalueDecompositionSymmetric<DataType::F32>>(
      lapack_ptr("ssyevd"));
  AssignKernelFn<EigenvalueDecompositionSymmetric<DataType::F64>>(
      lapack_ptr("dsyevd"));
  AssignKernelFn<EigenvalueDecompositionHermitian<DataType::C64>>(
      lapack_ptr("cheevd"));
  AssignKernelFn<EigenvalueDecompositionHermitian<DataType::C128>>(
      lapack_ptr("zheevd"));

  AssignKernelFn<EigenvalueDecomposition<DataType::F32>>(lapack_ptr("sgeev"));
  AssignKernelFn<EigenvalueDecomposition<DataType::F64>>(lapack_ptr("dgeev"));
  AssignKernelFn<EigenvalueDecompositionComplex<DataType::C64>>(
      lapack_ptr("cgeev"));
  AssignKernelFn<EigenvalueDecompositionComplex<DataType::C128>>(
      lapack_ptr("zgeev"));

  AssignKernelFn<SchurDecomposition<DataType::F32>>(lapack_ptr("sgees"));
  AssignKernelFn<SchurDecomposition<DataType::F64>>(lapack_ptr("dgees"));
  AssignKernelFn<SchurDecompositionComplex<DataType::C64>>(lapack_ptr("cgees"));
  AssignKernelFn<SchurDecompositionComplex<DataType::C128>>(
      lapack_ptr("zgees"));

  AssignKernelFn<HessenbergDecomposition<DataType::F32>>(lapack_ptr("sgehrd"));
  AssignKernelFn<HessenbergDecomposition<DataType::F64>>(lapack_ptr("dgehrd"));
  AssignKernelFn<HessenbergDecomposition<DataType::C64>>(lapack_ptr("cgehrd"));
  AssignKernelFn<HessenbergDecomposition<DataType::C128>>(lapack_ptr("zgehrd"));

  AssignKernelFn<TridiagonalReduction<DataType::F32>>(lapack_ptr("ssytrd"));
  AssignKernelFn<TridiagonalReduction<DataType::F64>>(lapack_ptr("dsytrd"));
  AssignKernelFn<TridiagonalReduction<DataType::C64>>(lapack_ptr("chetrd"));
  AssignKernelFn<TridiagonalReduction<DataType::C128>>(lapack_ptr("zhetrd"));

  initialized = true;
}

nb::dict Registrations() {
  nb::dict dict;

  dict["blas_strsm"] = EncapsulateFunction(blas_strsm);
  dict["blas_dtrsm"] = EncapsulateFunction(blas_dtrsm);
  dict["blas_ctrsm"] = EncapsulateFunction(blas_ctrsm);
  dict["blas_ztrsm"] = EncapsulateFunction(blas_ztrsm);
  dict["lapack_sgetrf"] = EncapsulateFunction(lapack_sgetrf);
  dict["lapack_dgetrf"] = EncapsulateFunction(lapack_dgetrf);
  dict["lapack_cgetrf"] = EncapsulateFunction(lapack_cgetrf);
  dict["lapack_zgetrf"] = EncapsulateFunction(lapack_zgetrf);
  dict["lapack_sgeqrf"] = EncapsulateFunction(lapack_sgeqrf);
  dict["lapack_dgeqrf"] = EncapsulateFunction(lapack_dgeqrf);
  dict["lapack_cgeqrf"] = EncapsulateFunction(lapack_cgeqrf);
  dict["lapack_zgeqrf"] = EncapsulateFunction(lapack_zgeqrf);
  dict["lapack_sorgqr"] = EncapsulateFunction(lapack_sorgqr);
  dict["lapack_dorgqr"] = EncapsulateFunction(lapack_dorgqr);
  dict["lapack_cungqr"] = EncapsulateFunction(lapack_cungqr);
  dict["lapack_zungqr"] = EncapsulateFunction(lapack_zungqr);
  dict["lapack_spotrf"] = EncapsulateFunction(lapack_spotrf);
  dict["lapack_dpotrf"] = EncapsulateFunction(lapack_dpotrf);
  dict["lapack_cpotrf"] = EncapsulateFunction(lapack_cpotrf);
  dict["lapack_zpotrf"] = EncapsulateFunction(lapack_zpotrf);
  dict["lapack_sgesdd"] = EncapsulateFunction(lapack_sgesdd);
  dict["lapack_dgesdd"] = EncapsulateFunction(lapack_dgesdd);
  dict["lapack_cgesdd"] = EncapsulateFunction(lapack_cgesdd);
  dict["lapack_zgesdd"] = EncapsulateFunction(lapack_zgesdd);
  dict["lapack_ssyevd"] = EncapsulateFunction(lapack_ssyevd);
  dict["lapack_dsyevd"] = EncapsulateFunction(lapack_dsyevd);
  dict["lapack_cheevd"] = EncapsulateFunction(lapack_cheevd);
  dict["lapack_zheevd"] = EncapsulateFunction(lapack_zheevd);
  dict["lapack_sgeev"] = EncapsulateFunction(lapack_sgeev);
  dict["lapack_dgeev"] = EncapsulateFunction(lapack_dgeev);
  dict["lapack_cgeev"] = EncapsulateFunction(lapack_cgeev);
  dict["lapack_zgeev"] = EncapsulateFunction(lapack_zgeev);
  dict["lapack_sgees"] = EncapsulateFunction(lapack_sgees);
  dict["lapack_dgees"] = EncapsulateFunction(lapack_dgees);
  dict["lapack_cgees"] = EncapsulateFunction(lapack_cgees);
  dict["lapack_zgees"] = EncapsulateFunction(lapack_zgees);
  dict["lapack_sgehrd"] = EncapsulateFunction(lapack_sgehrd);
  dict["lapack_dgehrd"] = EncapsulateFunction(lapack_dgehrd);
  dict["lapack_cgehrd"] = EncapsulateFunction(lapack_cgehrd);
  dict["lapack_zgehrd"] = EncapsulateFunction(lapack_zgehrd);
  dict["lapack_ssytrd"] = EncapsulateFunction(lapack_ssytrd);
  dict["lapack_dsytrd"] = EncapsulateFunction(lapack_dsytrd);
  dict["lapack_chetrd"] = EncapsulateFunction(lapack_chetrd);
  dict["lapack_zhetrd"] = EncapsulateFunction(lapack_zhetrd);

  return dict;
}

NB_MODULE(_lapack, m) {
  // Populates the LAPACK kernels from scipy on first call.
  m.def("initialize", GetLapackKernelsFromScipy);

  m.def("registrations", &Registrations);
  m.def("lapack_sgeqrf_workspace",
        &QrFactorization<DataType::F32>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_dgeqrf_workspace",
        &QrFactorization<DataType::F64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_cgeqrf_workspace",
        &QrFactorization<DataType::C64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_zgeqrf_workspace",
        &QrFactorization<DataType::C128>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_sorgqr_workspace",
        &OrthogonalQr<DataType::F32>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_dorgqr_workspace",
        &OrthogonalQr<DataType::F64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_cungqr_workspace",
        &OrthogonalQr<DataType::C64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_zungqr_workspace",
        &OrthogonalQr<DataType::C128>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("gesdd_iwork_size", &svd::GetIntWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("sgesdd_work_size", &GesddGetWorkspaceSize<DataType::F32>, nb::arg("m"),
        nb::arg("n"), nb::arg("job_opt_compute_uv"),
        nb::arg("job_opt_full_matrices"));
  m.def("dgesdd_work_size", &GesddGetWorkspaceSize<DataType::F64>, nb::arg("m"),
        nb::arg("n"), nb::arg("job_opt_compute_uv"),
        nb::arg("job_opt_full_matrices"));
  // TODO(paruzelp): Rename to gesdd_rwork_size or supply all types
  m.def("cgesdd_rwork_size", &GesddGetRealWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("compute_uv"));
  m.def("cgesdd_work_size", &GesddGetWorkspaceSize<DataType::C64>, nb::arg("m"),
        nb::arg("n"), nb::arg("job_opt_compute_uv"),
        nb::arg("job_opt_full_matrices"));
  m.def("zgesdd_work_size", &GesddGetWorkspaceSize<DataType::C128>,
        nb::arg("m"), nb::arg("n"), nb::arg("job_opt_compute_uv"),
        nb::arg("job_opt_full_matrices"));
  m.def("syevd_work_size", BoundWithEigvecs<eig::GetIntWorkspaceSize>,
        nb::arg("n"));
  m.def("syevd_iwork_size", BoundWithEigvecs<eig::GetIntWorkspaceSize>,
        nb::arg("n"));
  m.def("heevd_work_size", BoundWithEigvecs<eig::GetComplexWorkspaceSize>,
        nb::arg("n"));
  m.def("heevd_rwork_size", BoundWithEigvecs<eig::GetRealWorkspaceSize>,
        nb::arg("n"));

  m.def("lapack_sgehrd_workspace",
        &HessenbergDecomposition<DataType::F32>::GetWorkspaceSize,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_dgehrd_workspace",
        &HessenbergDecomposition<DataType::F64>::GetWorkspaceSize,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_cgehrd_workspace",
        &HessenbergDecomposition<DataType::C64>::GetWorkspaceSize,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_zgehrd_workspace",
        &HessenbergDecomposition<DataType::C128>::GetWorkspaceSize,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_ssytrd_workspace",
        &TridiagonalReduction<DataType::F32>::GetWorkspaceSize, nb::arg("lda"),
        nb::arg("n"));
  m.def("lapack_dsytrd_workspace",
        &TridiagonalReduction<DataType::F64>::GetWorkspaceSize, nb::arg("lda"),
        nb::arg("n"));
  m.def("lapack_chetrd_workspace",
        &TridiagonalReduction<DataType::C64>::GetWorkspaceSize, nb::arg("lda"),
        nb::arg("n"));
  m.def("lapack_zhetrd_workspace",
        &TridiagonalReduction<DataType::C128>::GetWorkspaceSize, nb::arg("lda"),
        nb::arg("n"));
}

}  // namespace
}  // namespace jax
