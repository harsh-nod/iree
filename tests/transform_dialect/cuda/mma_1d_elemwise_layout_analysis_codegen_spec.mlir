// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):

  // Step 1. Find the fill, matmul and generic ops
  // ===========================================================================
  %fills = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %matmuls = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %fill0 = transform.split_handles %fills in [1] : (!pdl.operation) -> (!pdl.operation)
  %matmul0, %matmul1 = transform.split_handles %matmuls in [2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %newMax, %subExp, %scaleOldSum, %newSum, %softmax, %scaleAcc = transform.split_handles %generics in [6]
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

  // Step 2. Tile the last op and fuse the rest
  // ===========================================================================
  %forall_grid, %grid_reduction =
  transform.structured.tile_to_forall_op %scaleAcc tile_sizes [16] ( mapping = [#gpu.block<x>] )
  transform.structured.fuse_into_containing_op %matmul1 into %forall_grid
  transform.structured.fuse_into_containing_op %softmax into %forall_grid
  transform.structured.fuse_into_containing_op %newSum into %forall_grid
  transform.structured.fuse_into_containing_op %scaleOldSum into %forall_grid
  transform.structured.fuse_into_containing_op %subExp into %forall_grid
  transform.structured.fuse_into_containing_op %newMax into %forall_grid
  transform.structured.fuse_into_containing_op %matmul0 into %forall_grid
  transform.structured.fuse_into_containing_op %fill0 into %forall_grid

  // Step 3. Vectorize
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func { rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
  %func_3 = transform.structured.vectorize %func

  // Step 4. Bufferize
  // ===========================================================================
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
  transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [4, 8, 1] : (!pdl.operation) -> ()

  // Step 7. Do layout analysis and lower to mma
  // ===========================================================================
  %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %reordered_func = transform.iree.reorder_transpose %func_10 : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %reordered_func { cse } : (!pdl.operation) -> ()
  %func_11 = transform.iree.layout_analysis_and_distribution %reordered_func : (!pdl.operation) -> (!pdl.operation)
}
