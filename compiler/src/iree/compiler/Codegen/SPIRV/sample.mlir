func.func @conv_dispatch0() {
    // G and GT constants should be promoted to shared memory    
    %cst = arith.constant dense_resource<__elided__> : tensor<4x8xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<8x4xf32>
    %0 = bufferization.to_memref %cst_1 : memref<8x4xf32, 3>
    %1 = bufferization.to_memref %cst : memref<4x8xf32, 3>
    // Allocated (in shmem) and fill padded kernel tensor with zeros 
    %alloc_3 = memref.alloc() {alignment = 128 : i64} : memref<4x4xf32, 3>
    %cst0 = arith.constant dense<0.000000e+00> : vector<4xf32>
    vector.transfer_write %cst0, %alloc_3[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x4xf32, 3>
    vector.transfer_write %cst0, %alloc_3[%c1, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x4xf32, 3>
    vector.transfer_write %cst0, %alloc_3[%c2, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x4xf32, 3>
    vector.transfer_write %cst0, %alloc_3[%c3, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x4xf32, 3>
    // Allocate interim result buffer in shmem (this will be filled before the matmul)
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<4x8xf32, 3>
    // Define workgroup ids
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
    %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %subview = memref.subview %2[0, 0, %4, %5] [3, 3, 32, 32] [1, 1, 1, 1] : memref<3x3x1280x1280xf32> to memref<3x3x32x32xf32, strided<[4915200, 1638400, 1280, 1], offset: ?>>
    %subview_3 = memref.subview %3[0, 0, %4, %5] [8, 8, 32, 32] [1, 1, 1, 1] : memref<8x8x1280x1280xf32> to memref<8x8x32x32xf32, strided<[13107200, 1638400, 1280, 1], offset: ?>>
    // Define gpu ids
    %6 = gpu.thread_id  x
    %7 = gpu.thread_id  y
    %8 = gpu.thread_id  z

    // Promote inputs and outputs
     %11 = vector.transfer_read %subview_7[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x4xf32, strided<[1638400, 1280, 1], offset: ?>>, vector<1x1x4xf32>
    vector.transfer_write %11, %subview_8[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x1x4xf32>, memref<1x1x4xf32, strided<[2048, 128, 1], offset: ?>, 3>

    gpu.barrier

    // Compute on promoted inputs and outputs
    linalg.fill ins(%zero : f32) outs(%interm : memref<4x8xf32, 3>) 
    linalg.matmul ins(%subview_input_slice, %1 : memref<4x4xf32, 3>, memref<4x8xf32, 3>) outs(%interm : memref<4x8xf32, 3>)

    linalg.fill ins(%zero : f32) outs(%subview_out_slice : memref<8x8xf32, 3>)
    linalg.matmul ins(%0, %interim : memref<8x4xf32, 3>, memref<4x8xf32, 3>) outs(%subview_out_slice : memref<4x8xf32, 3>)

}