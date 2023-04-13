// RUN: iree-opt %s

transform.sequence failures(propagate){
^bb1(%variant_op: !pdl.operation):
  %ops = transform.structured.match ops{["linalg.fill", "linalg.matmul", "linalg.generic"]}
    in %variant_op : (!pdl.operation) -> !pdl.operation

  %fill0, %matmul0, %newMax, %subExp, %scaleOldSum, %newSum, %softmax, %matmul1, %scaleAcc =
    transform.split_handles %ops in [9]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation,
                             !pdl.operation, !pdl.operation, !pdl.operation,
                             !pdl.operation, !pdl.operation)

  %region_op = transform.iree.wrap_in_dispatch_region %scaleAcc { generateWorkload = false }

  %all_but_last = transform.merge_handles %fill0, %matmul0, %newMax, %subExp, %scaleOldSum, %newSum, %softmax, %matmul1 : !pdl.operation
  %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %all_but_last into %region_op

  %empty = transform.structured.match ops{["tensor.empty"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %region_op_3 = transform.iree.move_preceding_op_into_dispatch_region %empty into %region_op_2
  transform.iree.region_to_workgroups %region_op_3
}
