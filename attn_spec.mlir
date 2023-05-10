transform.sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):

    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!pdl.operation) -> !pdl.operation

    // Tile and distribute to workgroups
    // ==========================================
    %forall_grid, %tiled_attention =
    transform.structured.tile_to_forall_op %attention tile_sizes [1, 128]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] )
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!pdl.operation) -> ()

    // Tile and decompose attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    %outer_loop, %output_fill, %max_fill, %sum_fill, %mid_loop, %inner_loop, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %reduce_sum, %reciprocal_sum, %update,
    %softmax, %scale_acc, %second_matmul, %trunc = tile_and_decompose_attention %attention2 :
       (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

    // Distribute fills
    // ==========================================
    %fills = transform.merge_handles %output_fill, %max_fill, %sum_fill : !pdl.operation
    %fill_grid, %tiled_fill = transform.structured.tile_to_forall_op %fills tile_sizes[32] (mapping = [#gpu.warp<x>])

    // Vectorize function
    // ==========================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
    %func_3 = transform.structured.vectorize %func

    // Bufferization
    // ==========================================
    transform.iree.apply_patterns %func_3
      { fold_reassociative_reshapes, canonicalization, tiling_canonicalization, cse } : (!pdl.operation) -> ()
    transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
    transform.iree.apply_patterns %func_3 { erase_unnecessary_tensor_operands } : (!pdl.operation) -> ()
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!pdl.operation) -> (!pdl.operation)
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %func_2 {  prepare_vector_to_mma } : (!pdl.operation) -> ()

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    transform.iree.forall_to_workgroup %func_7 : (!pdl.operation) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [4, 8, 4] warp_dims = [4, 1, 1] : (!pdl.operation) -> ()

    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %func_8 { fold_memref_aliases } : (!pdl.operation) -> ()
    transform.iree.apply_patterns %func_8 { cse } : (!pdl.operation) -> ()
    transform.iree.apply_patterns %func_8 { canonicalization } : (!pdl.operation) -> ()
    transform.iree.apply_buffer_optimizations %func_8 : (!pdl.operation) -> ()

    // Step 7. Do layout analysis and lower to mma
    // ===========================================================================
    %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    %reordered_func = transform.iree.reorder_transpose %func_10 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %reordered_func { cse } : (!pdl.operation) -> ()
    %reordered_func2 = transform.iree.reorder_transpose %reordered_func : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %reordered_func2 { cse } : (!pdl.operation) -> ()
    %func_x = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %func_x {  prepare_vector_to_mma, cse } : (!pdl.operation) -> ()
    %func_y = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    %preproc_func = transform.iree.eliminate_shmem_trip %func_y : (!pdl.operation) -> !pdl.operation
    %func_11 = transform.iree.layout_analysis_and_distribution %preproc_func : (!pdl.operation) -> (!pdl.operation)
}
