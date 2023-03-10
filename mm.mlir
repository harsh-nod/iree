func.func @matmul(%lhs : tensor<16x16xf16>, %rhs : tensor<16x8xf16>) -> tensor<16x8xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = tensor.empty() : tensor<16x8xf16>
  %1 = linalg.fill ins(%c0 : f16) outs(%0 : tensor<16x8xf16>) -> tensor<16x8xf16>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x8xf16>)
      outs(%1 : tensor<16x8xf16>) -> tensor<16x8xf16>
  return %2 : tensor<16x8xf16>
}
