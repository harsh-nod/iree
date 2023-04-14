func.func @attention(%query: tensor<20x1024x64xf32>, %key: tensor<20x1024x64xf32>, %value: tensor<20x1024x64xf32>) -> tensor<20x1024x64xf32> {
  %0 = tensor.empty() : tensor<20x1024x64xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<20x1024x64xf32>, tensor<20x1024x64xf32>, tensor<20x1024x64xf32>) outs(%0 : tensor<20x1024x64xf32>) -> tensor<20x1024x64xf32>
  return %1 : tensor<20x1024x64xf32>
}
