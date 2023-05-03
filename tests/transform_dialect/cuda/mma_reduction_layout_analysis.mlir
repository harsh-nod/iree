#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @matmul_reduction(%lhs : tensor<16x16xf32>, %rhs : tensor<8x16xf32>) -> tensor<16x8xf32> {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant -1.0e+04 : f32
  %acc = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%c1 : f32) outs(%acc : tensor<16xf32>) -> tensor<16xf32>
  %0 = tensor.empty() : tensor<16x8xf32>
  %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<16x8xf32>) -> tensor<16x8xf32>
  %2 = linalg.matmul_transpose_b ins(%lhs, %rhs : tensor<16x16xf32>, tensor<8x16xf32>)
      outs(%1 : tensor<16x8xf32>) -> tensor<16x8xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
        ins(%2 : tensor<16x8xf32>) outs(%init : tensor<16xf32>) {
        ^bb0(%in: f32, %out: f32):
          %20 = arith.maxf %in, %out : f32
          linalg.yield %20 : f32
        } -> tensor<16xf32>
  %8 = linalg.generic {indexing_maps = [#map1, #map], iterator_types=["parallel", "parallel"]}
        ins(%6 : tensor<16xf32>) outs(%0 : tensor<16x8xf32>) {
        ^bb0(%in: f32,  %out: f32):
          linalg.yield %in : f32
        } -> tensor<16x8xf32>
  return %8 : tensor<16x8xf32>
}

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-hal-cuda-llvm-target-arch=sm_80 \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-flow-dispatch-use-transform-dialect=%p/mma_reduction_layout_analysis_dispatch_spec.mlir \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/mma_reduction_layout_analysis_codegen_spec.mlir | \
// RUN: iree-run-module --function=matmul_reduction --device=cuda \
// RUN: --input="16x16xf32=[[1.0,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2.0,2.125,2.25,2.375,2.5,2.625,2.75,2.875],[3.0,3.125,3.25,3.375,3.5,3.625,3.75,3.875,4.0,4.125,4.25,4.375,4.5,4.625,4.75,4.875],[5.0,5.125,5.25,5.375,5.5,5.625,5.75,5.875,6.0,6.125,6.25,6.375,6.5,6.625,6.75,6.875],[7.0,7.125,7.25,7.375,7.5,7.625,7.75,7.875,8.0,8.125,8.25,8.375,8.5,8.625,8.75,8.875],[9.0,9.125,9.25,9.375,9.5,9.625,9.75,9.875,10.0,10.125,10.25,10.375,10.5,10.625,10.75,10.875],[11.0,11.125,11.25,11.375,11.5,11.625,11.75,11.875,12.0,12.125,12.25,12.375,12.5,12.625,12.75,12.875],[13.0,13.125,13.25,13.375,13.5,13.625,13.75,13.875,14.0,14.125,14.25,14.375,14.5,14.625,14.75,14.875],[15.0,15.125,15.25,15.375,15.5,15.625,15.75,15.875,16.0,16.125,16.25,16.375,16.5,16.625,16.75,16.875],[17.0,17.125,17.25,17.375,17.5,17.625,17.75,17.875,18.0,18.125,18.25,18.375,18.5,18.625,18.75,18.875],[19.0,19.125,19.25,19.375,19.5,19.625,19.75,19.875,20.0,20.125,20.25,20.375,20.5,20.625,20.75,20.875],[21.0,21.125,21.25,21.375,21.5,21.625,21.75,21.875,22.0,22.125,22.25,22.375,22.5,22.625,22.75,22.875],[23.0,23.125,23.25,23.375,23.5,23.625,23.75,23.875,24.0,24.125,24.25,24.375,24.5,24.625,24.75,24.875],[25.0,25.125,25.25,25.375,25.5,25.625,25.75,25.875,26.0,26.125,26.25,26.375,26.5,26.625,26.75,26.875],[27.0,27.125,27.25,27.375,27.5,27.625,27.75,27.875,28.0,28.125,28.25,28.375,28.5,28.625,28.75,28.875],[29.0,29.125,29.25,29.375,29.5,29.625,29.75,29.875,30.0,30.125,30.25,30.375,30.5,30.625,30.75,30.875],[31.0,31.125,31.25,31.375,31.5,31.625,31.75,31.875,32.0,32.125,32.25,32.375,32.5,32.625,32.75,32.875]]" \
// RUN: --input="8x16xf32=[[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16][1.125 2.125 3.125 4.125 5.125 6.125 7.125 8.125 9.125 10.125 11.125 12.125 13.125 14.125 15.125 16.125][1.25 2.25 3.25 4.25 5.25 6.25 7.25 8.25 9.25 10.25 11.25 12.25 13.25 14.25 15.25 16.25][1.375 2.375 3.375 4.375 5.375 6.375 7.375 8.375 9.375 10.375 11.375 12.375 13.375 14.375 15.375 16.375][1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5 12.5 13.5 14.5 15.5 16.5][1.625 2.625 3.625 4.625 5.625 6.625 7.625 8.625 9.625 10.625 11.625 12.625 13.625 14.625 15.625 16.625][1.75 2.75 3.75 4.75 5.75 6.75 7.75 8.75 9.75 10.75 11.75 12.75 13.75 14.75 15.75 16.75][1.875 2.875 3.875 4.875 5.875 6.875 7.875 8.875 9.875 10.875 11.875 12.875 13.875 14.875 15.875 16.875]]" |\
// RUN: FileCheck %s --check-prefix=EXEC

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 16x8xf32=[333 333 333 333 333 333 333 333][633 633 633 633 633 633 633 633][933 933 933 933 933 933 933 933][1233 1233 1233 1233 1233 1233 1233 1233][1533 1533 1533 1533 1533 1533 1533 1533][1833 1833 1833 1833 1833 1833 1833 1833][2134 2134 2134 2134 2134 2134 2134 2134][2434 2434 2434 2434 2434 2434 2434 2434][2734 2734 2734 2734 2734 2734 2734 2734][3034 3034 3034 3034 3034 3034 3034 3034][3334 3334 3334 3334 3334 3334 3334 3334][3634 3634 3634 3634 3634 3634 3634 3634][3934 3934 3934 3934 3934 3934 3934 3934][4232 4232 4232 4232 4232 4232 4232 4232][4532 4532 4532 4532 4532 4532 4532 4532][4832 4832 4832 4832 4832 4832 4832 4832]
