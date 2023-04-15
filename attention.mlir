func.func @attention(%query: tensor<20x1024x64xf16>, %key: tensor<20x1024x64xf16>, %value: tensor<20x1024x64xf16>) -> tensor<20x1024x64xf16> {
  %0 = tensor.empty() : tensor<20x1024x64xf16>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<20x1024x64xf16>, tensor<20x1024x64xf16>, tensor<20x1024x64xf16>) outs(%0 : tensor<20x1024x64xf16>) -> tensor<20x1024x64xf16>
  return %1 : tensor<20x1024x64xf16>
}
