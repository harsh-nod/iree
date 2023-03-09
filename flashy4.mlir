transform.structured.canonicalized_sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):

    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op

    // Tile and distribute to workgroups
    // ==========================================
    %foreach_thread_grid, %tiled_attention =
    transform.iree.tile_to_foreach_thread_and_workgroup_count_region %attention tile_sizes [1, 128]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] )

    // Tile and decompose attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op
    %outer_loop, %mid_loop, %inner_loop, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %reduce_sum, %update,
    %softmax, %scale_acc, %second_matmul = transform.iree.tile_and_decompose_attention %attention2 :
       (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

    //%foreach_thread_a, %tiled_op_a = transform.structured.tile_to_foreach_thread_op %update num_threads [32] (mapping = [#gpu.thread<x>])
    //%foreach_thread_b, %tiled_op_b = transform.structured.tile_to_foreach_thread_op %reduce_sum num_threads [32] (mapping = [#gpu.thread<x>])
    //%foreach_thread_c, %tiled_op_c = transform.structured.tile_to_foreach_thread_op %partial_softmax num_threads [32] (mapping = [#gpu.thread<x>])
    //%foreach_thread_d, %tiled_op_d = transform.structured.tile_to_foreach_thread_op %softmax num_threads [32] (mapping = [#gpu.thread<x>])
    //%foreach_thread_e, %tiled_op_e = transform.structured.tile_to_foreach_thread_op %reduce_max num_threads [32] (mapping = [#gpu.thread<x>])

    //transform.structured.fuse_into_containing_op %reduce_sum into %foreach_thread
    //transform.structured.fuse_into_containing_op %partial_softmax into %foreach_thread
    //transform.structured.fuse_into_containing_op %reduce_max into %foreach_thread

    //%foreach_thread_2, %tiled_op_2 = transform.structured.tile_to_foreach_thread_op %scale_acc num_threads [32] (mapping = [#gpu.thread<x>])

    // Tile first matmul
    //%foreach_thread_3, %tiled_matmul = transform.structured.tile_to_foreach_thread_op %first_matmul num_threads [1] ( mapping = [#gpu.block<x>] )
    //transform.structured.fuse_into_containing_op %fill_op into %foreach_thread_3

    // Tile second matmul
    //%foreach_thread_4, %tiled_matmul_2 = transform.structured.tile_to_foreach_thread_op %second_matmul num_threads [1] ( mapping = [#gpu.block<x>] )

    // Vectorize function
    %func = transform.structured.match ops{["func.func"]} in %variant_op
    %funcx = transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector }
    transform.structured.vectorize %funcx

    // Bufferization
    %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op_2
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3
    transform.iree.erase_hal_descriptor_type_from_memref %memref_func

    //// Convert vector to mma
    //%func2 = transform.structured.match ops{["func.func"]} in %variant_op_3
    //transform.iree.vector.vector_to_mma_conversion %func2

    //// Map to GPU thread blocks
    //%func3 = transform.structured.match ops{["func.func"]} in %variant_op_3
    //%func4 = transform.iree.foreach_thread_to_workgroup %func3
    //%func5 = transform.iree.map_nested_foreach_thread_to_gpu_threads %func4 {workgroup_size = [32, 1, 1]}

    //// Step 6. Post-bufferization vector distribution with rank-reduction.
    //// ===================================================================
    //%end_func = transform.structured.match ops{["func.func"]} in %variant_op_3
    //%end_func_2 = transform.iree.apply_patterns %end_func { rank_reducing_linalg, rank_reducing_vector, fold_memref_aliases }
}
