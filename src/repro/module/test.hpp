#include <Kokkos_Core.hpp>
#include <cstdio>

// Taken from
// https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/04/Solution/exercise_4_solution.cpp

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif
#ifdef KOKKOS_ENABLE_HIP
#define MemSpace Kokkos::Experimental::HIPSpace
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
#define MemSpace Kokkos::OpenMPTargetSpace
#endif

#ifndef MemSpace
#define MemSpace Kokkos::HostSpace
#endif

using ExecSpace = MemSpace::execution_space;
using range_policy = Kokkos::RangePolicy<ExecSpace>;

// Allocate y, x vectors and Matrix A on device.
typedef Kokkos::View<double *, Kokkos::LayoutLeft, MemSpace> ViewVectorType;
typedef Kokkos::View<double **, Kokkos::LayoutLeft, MemSpace> ViewMatrixType;

void bench(int N = 10, int M = 10, int S = 10, int nrepeat = 10) {
  ViewVectorType y("y", N);
  ViewVectorType x("x", M);
  ViewMatrixType A("A", N, M);

  // Create host mirrors of device views.
  ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view(y);
  ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view(x);
  ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view(A);

  // Initialize y vector on host.
  for (int i = 0; i < N; ++i) {
    h_y(i) = 1;
  }

  // Initialize x vector on host.
  for (int i = 0; i < M; ++i) {
    h_x(i) = 1;
  }

  // Initialize A matrix on host.
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      h_A(j, i) = 1;
    }
  }

  // Deep copy host views to device views.
  Kokkos::deep_copy(y, h_y);
  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(A, h_A);

  // Timer products.
  Kokkos::Timer timer;

  for (int repeat = 0; repeat < nrepeat; repeat++) {
    // Application: <y,Ax> = y^T*A*x
    double result = 0;

    Kokkos::parallel_reduce(
        "yAx", range_policy(0, N),
        KOKKOS_LAMBDA(int j, double &update) {
          double temp2 = 0;

          for (int i = 0; i < M; ++i) {
            temp2 += A(j, i) * x(i);
          }

          update += y(j) * temp2;
        },
        result);

    // Output result.
    if (repeat == (nrepeat - 1)) {
      printf("  Computed result for %d x %d is %lf\n", N, M, result);
    }

    const double solution = (double)N * (double)M;

    if (result != solution) {
      printf("  Error: result( %lf ) != solution( %lf )\n", result, solution);
    }
  }

  // Calculate time.
  double time = timer.seconds();

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double(sizeof(double) * (M + M * N + N));

  // Print results (problem size, time and bandwidth in GB/s).
  printf("  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) "
         "bandwidth( %g GB/s )\n",
         N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);
}
