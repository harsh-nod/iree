// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

namespace {

// Computes a reduction along the rows of a 2d tensor of shape MxN
// to produce a tensor of shape M
template <typename T>
static Value computeRowwiseReduction(Value a, Value output, Location loc,
                                     OpBuilder &builder,
                                     SmallVectorImpl<Operation *> &ops) {
  SmallVector<utils::IteratorType> iteratorTypes{
      utils::IteratorType::parallel, utils::IteratorType::reduction};
  AffineMap id = AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{id, rowMap};
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), a, output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<T>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value computePartialSoftmax(Value qkTranspose, Value currentMax,
                                   Location loc, OpBuilder &builder,
                                   SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, qkTranspose.getType(), ValueRange{currentMax}, qkTranspose,
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[1], args[0]);
        Value result = b.create<math::Exp2Op>(loc, diff);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value computeScaleFactor(Value oldMax, Value newMax,
                                Location loc, OpBuilder &builder,
                                SmallVectorImpl<Operation *> &ops) {
  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  auto identityMap = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps(2, identityMap);
  SmallVector<Type> resultTypes{oldMax.getType()};
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, resultTypes, ValueRange{newMax}, ValueRange{oldMax},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[1], args[0]);
        Value weight = b.create<math::Exp2Op>(loc, diff);
        b.create<linalg::YieldOp>(loc, ValueRange{weight});
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value updateAndScale(Value scaleFactor, Value oldSum,
                            Location loc, OpBuilder &builder,
                            SmallVectorImpl<Operation *> &ops) {
  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  auto identityMap = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps(2, identityMap);
  SmallVector<Type> resultTypes{oldSum.getType()};
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, resultTypes, ValueRange{scaleFactor}, ValueRange{oldSum},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value scaledOldSum = b.create<arith::MulFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, ValueRange{scaledOldSum});
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value scalePartialSoftmax(Value softmax, Value inverseNewSum, Location loc,
                                 OpBuilder &builder,
                                 SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, softmax.getType(), ValueRange{inverseNewSum}, softmax, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::MulFOp>(loc, args[1], args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value computeReciprocal(Value x, Location loc,
                              OpBuilder &builder,
                              SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(1,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, x.getType(), ValueRange{},
      x, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(args[0].getType(), 1.0));
        Value result = b.create<arith::DivFOp>(loc, one, args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value applyFinalScaling(Value result, Value newSum, Location loc, OpBuilder &builder,
                               SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps= {rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, result.getType(), ValueRange{newSum}, result, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value one = b.create<arith::ConstantOp>(
            loc, b.getFloatAttr(args[0].getType(), 1.0));
        Value reciprocal = b.create<arith::DivFOp>(loc, one, args[0]);
        Value result = b.create<arith::MulFOp>(loc, reciprocal, args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value scaleAccumulator(Value accumulator, Value scaleFactor,
                              Location loc,
                              OpBuilder &builder,
                              SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(2, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{rowMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, accumulator.getType(), ValueRange{scaleFactor},
      accumulator, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::MulFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static Value computeQKTranspose(Value query, Value key, Value output,
                                Value zero, Location loc, OpBuilder &builder,
                                SmallVectorImpl<Operation *> &ops) {
  auto fillOp = builder.create<linalg::FillOp>(loc, ValueRange{zero}, output);
  ops.push_back(fillOp);
  Value acc = fillOp.result();
  auto matmulOp = builder.create<linalg::MatmulTransposeBOp>(
      loc, output.getType(), ValueRange{query, key}, acc);
  ops.push_back(matmulOp);
  return matmulOp.getResult(0);
}

static std::tuple<Value, Value>
extractSlices(Value key, Value value,
              ArrayRef<int64_t> queryShape, ArrayRef<Value> ivs,
              OpFoldResult keyValueTileLength, OpFoldResult headDimension,
              Type elementType, Location loc, OpBuilder &builder) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> strides(queryShape.size(), one);
  SmallVector<OpFoldResult> sizes(queryShape.size(), one);
  SmallVector<OpFoldResult> offsets(queryShape.size(), zero);
  sizes[1] = keyValueTileLength;
  sizes[2] = headDimension;
  offsets[0] = ivs[0];
  offsets[1] = ivs[1];
  SmallVector<int64_t> tensorShape{queryShape[1], queryShape[2]};
  auto tensorType = RankedTensorType::get(tensorShape, elementType);
  Value keySlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, key, offsets, sizes, strides);
  Value valueSlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, value, offsets, sizes, strides);
  return std::make_tuple(keySlice, valueSlice);
}

static std::tuple<Value, Value, Value>
insertSlices(Value newResult, Value result, Value newMax, Value max,
             Value newSum, Value sum, ArrayRef<int64_t> queryShape,
             ArrayRef<Value> ivs, OpFoldResult sequenceTileLength,
             OpFoldResult headDimension, Location loc,
             OpBuilder &builder) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> strides(queryShape.size(), one);
  SmallVector<OpFoldResult> sizes(queryShape.size(), one);
  SmallVector<OpFoldResult> offsets(queryShape.size(), zero);
  sizes[1] = sequenceTileLength;
  sizes[2] = headDimension;
  offsets[0] = ivs[0];
  Value updatedAcc = newResult;
  offsets = SmallVector<OpFoldResult>(queryShape.size() - 2, zero);
  sizes = SmallVector<OpFoldResult>{sequenceTileLength};
  strides = SmallVector<OpFoldResult>(queryShape.size() - 2, one);
  Value updatedMax = builder.create<tensor::InsertSliceOp>(
      loc, newMax, max, offsets, sizes, strides);
  Value updatedSum = builder.create<tensor::InsertSliceOp>(
      loc, newSum, sum, offsets, sizes, strides);
  return std::make_tuple(updatedAcc, updatedMax, updatedSum);
}

static std::tuple<Value, Value, Value, Value>
extractSlicesInner(Value query, Value output, Value max, Value sum,
                   ArrayRef<int64_t> queryShape, int64_t tileSize,
                   ArrayRef<Value> ivs, OpFoldResult sequenceTileLength,
                   OpFoldResult headDimension,
                   Type elementType, Location loc, OpBuilder &builder) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> strides(queryShape.size(), one);
  SmallVector<OpFoldResult> sizes(queryShape.size(), one);
  SmallVector<OpFoldResult> offsets(queryShape.size(), zero);
  sizes[1] = sequenceTileLength;
  sizes[2] = headDimension;
  offsets[0] = ivs[0];
  offsets[1] = ivs[2];
  SmallVector<int64_t> tensorShape{tileSize, queryShape[2]};
  auto tensorType = RankedTensorType::get(tensorShape, elementType);
  Value querySlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, query, offsets, sizes, strides);

  offsets = SmallVector<OpFoldResult>(queryShape.size() - 1, zero);
  sizes = SmallVector<OpFoldResult>(queryShape.size() - 1, one);
  strides = SmallVector<OpFoldResult>(queryShape.size() - 1, one);
  offsets[0] = ivs[2];
  sizes[0] = sequenceTileLength;
  sizes[1] = headDimension;
  tensorType = RankedTensorType::get(tensorShape, builder.getF32Type());
  Value outputSlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, output, offsets, sizes, strides);

  offsets = SmallVector<OpFoldResult>(queryShape.size() - 2, zero);
  sizes = SmallVector<OpFoldResult>(queryShape.size() - 2, one);
  strides = SmallVector<OpFoldResult>(queryShape.size() - 2, one);
  offsets[0] = ivs[2];
  sizes[0] = sequenceTileLength;
  tensorShape = SmallVector<int64_t>{tileSize};
  tensorType = RankedTensorType::get(tensorShape, builder.getF32Type());
  Value maxSlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, max, offsets, sizes, strides);
  Value sumSlice = builder.create<tensor::ExtractSliceOp>(
      loc, tensorType, sum, offsets, sizes, strides);

  return std::make_tuple(querySlice, outputSlice, maxSlice, sumSlice);
}

static void insertSlicesInner(Value newResult, Value result, Value newMax,
                              Value max, Value newSum, Value sum,
                              ArrayRef<int64_t> queryShape, ArrayRef<Value> ivs,
                              OpFoldResult sequenceTileLength, OpFoldResult headDimension,
                              Location loc,
                              OpBuilder &builder) {
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> strides(queryShape.size() - 1, one);
  SmallVector<OpFoldResult> sizes(queryShape.size() - 1, one);
  SmallVector<OpFoldResult> offsets(queryShape.size() - 1, zero);
  sizes[0] = sequenceTileLength;
  sizes[1] = headDimension;
  offsets[0] = ivs[2];
  builder.create<tensor::ParallelInsertSliceOp>(loc, newResult, result, offsets,
                                                sizes, strides);
  offsets = SmallVector<OpFoldResult>(queryShape.size() - 2, zero);
  offsets[0] = ivs[2];
  sizes = SmallVector<OpFoldResult>{sequenceTileLength};
  strides = SmallVector<OpFoldResult>(queryShape.size() - 2, one);
  builder.create<tensor::ParallelInsertSliceOp>(loc, newMax, max, offsets,
                                                sizes, strides);
  builder.create<tensor::ParallelInsertSliceOp>(loc, newSum, sum, offsets,
                                                sizes, strides);
  return;
}

static scf::LoopNest createLoopNest(SmallVectorImpl<Value> &ivs, Value lb,
                                    Value step, Value ub, ValueRange args,
                                    Location loc, OpBuilder &builder) {
  SmallVector<Value> lbs{lb};
  SmallVector<Value> steps{step};
  SmallVector<Value> ubs{ub};
  scf::LoopNest loopNest = scf::buildLoopNest(
      builder, loc, lbs, ubs, steps, args,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
          ValueRange iterArgs) -> scf::ValueVector { return iterArgs; });
  for (scf::ForOp loop : loopNest.loops)
    ivs.push_back(loop.getInductionVar());
  return loopNest;
}

Value extendToF32(Value input, Value output, OpBuilder &builder, Location loc) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(3, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(3, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(3,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{input},
      output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::ExtFOp>(loc, b.getF32Type(), args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

template <int T>
Value truncateToF16(Value input, Value output, OpBuilder &builder, Location loc, SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(T, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(T, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(T,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{input},
      output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::TruncFOp>(loc, b.getF16Type(), args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

template <int T>
Value truncateToF16DPS(Value output, OpBuilder &builder, Location loc, SmallVectorImpl<Operation *> &ops) {
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(T, builder.getContext());
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  // (d0, d1) -> (d0)
  auto rowMap = AffineMap::get(T, 0, {d0}, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(T,
                                                 utils::IteratorType::parallel);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{},
      output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::TruncFOp>(loc, b.getF16Type(), args[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);
  return genericOp.getResult(0);
}

static std::tuple<Value, Value, Value>
createAttentionBody(Value keySlice, Value valueSlice, Value querySlice,
                    Value outputSlice, Value maxSlice, Value sumSlice,
                    OpFoldResult tileSize,
                    OpFoldResult sequenceTileLength, OpFoldResult keyValueTileLength,
                    OpFoldResult headDimension,
                    Type elementType, SmallVectorImpl<Operation *> &ops,
                    Location loc, OpBuilder &builder) {

  // Compute matmul(q, transpose(k))
  Type f32Type = builder.getF32Type();
  Value zero =
      builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(f32Type));
  SmallVector<OpFoldResult> resultShape{tileSize, keyValueTileLength};
  Value emptySquare =
      builder.create<tensor::EmptyOp>(loc, resultShape, f32Type);
  Value qkTranspose = computeQKTranspose(querySlice, keySlice, emptySquare,
                                         zero, loc, builder, ops);

  // Compute current statistics
  Value newMax = computeRowwiseReduction<arith::MaxFOp>(qkTranspose, maxSlice,
                                                        loc, builder, ops);
  Value scaleFactor = computeScaleFactor(maxSlice, newMax, loc, builder, ops);
  Value scaledOldSum =
      updateAndScale(scaleFactor, sumSlice, loc, builder, ops);
  Value partialSoftmax =
      computePartialSoftmax(qkTranspose, newMax, loc, builder, ops);

  ArrayRef<int64_t> softmaxShape = partialSoftmax.getType().cast<ShapedType>().getShape();
  Value scratch = builder.create<tensor::EmptyOp>(loc, resultShape, builder.getF16Type());
  Value truncatedPartialSoftmax = truncateToF16<2>(partialSoftmax, scratch, builder, loc, ops);

  // Update accumulator
  Value scaledAcc = scaleAccumulator(outputSlice, scaleFactor, loc, builder, ops);

  // Compute matmul(softmax, v)
  auto matmulOp = builder.create<linalg::MatmulOp>(
      loc, scaledAcc.getType(), ValueRange{truncatedPartialSoftmax, valueSlice}, scaledAcc);
  ops.push_back(matmulOp);
  Value result = matmulOp.getResult(0);

  Value newSum = computeRowwiseReduction<arith::AddFOp>(
      partialSoftmax, scaledOldSum, loc, builder, ops);

  return std::make_tuple(result, newMax, newSum);
}

} // namespace

SmallVector<Operation *>
tileAndDecomposeAttention(IREE::LinalgExt::AttentionOp attnOp,
                          IRRewriter &rewriter) {
  SmallVector<Operation *> ops;
  Location loc = attnOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(attnOp);

  Value query = attnOp.getQuery();
  ShapedType queryType = attnOp.getQueryType();
  Type elementType = queryType.getElementType();
  ArrayRef<int64_t> queryShape = queryType.getShape();
  SmallVector<OpFoldResult> queryDimValues =
      tensor::createDimValues(rewriter, loc, query);
  OpFoldResult headDimension = queryDimValues[2];
  OpFoldResult sequenceTileLength = queryDimValues[1];
  OpFoldResult batchTileLength = queryDimValues[0];

  Value key = attnOp.getKey();
  Value value = attnOp.getValue();
  SmallVector<int64_t> keyShape{attnOp.getKeyType().getShape()};
  keyShape[1] = 64;
  auto keyValueTileLength = rewriter.getIndexAttr(keyShape[1]);
  SmallVector<OpFoldResult> keyDimValues =
      tensor::createDimValues(rewriter, loc, key);
  OpFoldResult sequenceLength = keyDimValues[1];

  bufferization::BufferizationOptions options;
  FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
                rewriter, loc, query, false, options, true);
  if (failed(ret)) {
    return {};
  }
  query = ret.value();

  Value output = attnOp.getOutput();
  ret = bufferization::allocateTensorForShapedValue(
                rewriter, loc, output, false, options, true);
  if (failed(ret)) {
    return {};
  }
  output = ret.value();

  // Construct first loop
  Value zeroValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value oneValue = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> ivs;
  scf::LoopNest firstLoopNest = createLoopNest(
      ivs, zeroValue, oneValue,
      getValueOrCreateConstantIndexOp(rewriter, loc, batchTileLength),
      ValueRange({output}), loc, rewriter);
  Value iterArg = firstLoopNest.loops.back().getRegionIterArg(0);
  ops.push_back(firstLoopNest.loops.back());

  OpBuilder::InsertionGuard guardFirstLoop(rewriter);
  rewriter.setInsertionPointToStart(firstLoopNest.loops.back().getBody());

  // Create output in fp32
  SmallVector<OpFoldResult> shape =
      tensor::createDimValues(rewriter, loc, iterArg);
  Type statType = rewriter.getF32Type();
  Value zeroF32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(statType));
  Value outputEmpty = rewriter.create<tensor::EmptyOp>(loc, SmallVector<OpFoldResult>{shape[1], shape[2]}, statType);
  auto outputFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32}, outputEmpty);
  Value outputF32 = outputFill.result();
  ops.push_back(outputFill);

  // Create max and sum statistics
  SmallVector<OpFoldResult> dims{sequenceTileLength};
  Value largeNegativeF32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(statType, -1.0e+30));
  Value max = rewriter.create<tensor::EmptyOp>(loc, dims, statType);
  auto maxFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{largeNegativeF32}, max);
  ops.push_back(maxFill);
  Value negativeMax = maxFill.result();
  Value sum = rewriter.create<tensor::EmptyOp>(loc, dims, statType);
  auto sumFill = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32}, sum);
  ops.push_back(sumFill);
  Value zeroSum = sumFill.result();

  // Construct second loop
  scf::LoopNest secondLoopNest = createLoopNest(
      ivs, zeroValue,
      getValueOrCreateConstantIndexOp(rewriter, loc, keyValueTileLength),
      getValueOrCreateConstantIndexOp(rewriter, loc, sequenceLength),
      ValueRange({outputF32, negativeMax, zeroSum}), loc, rewriter);
  ops.push_back(secondLoopNest.loops.back());

  Value iterArgResult = secondLoopNest.loops.back().getRegionIterArg(0);
  Value iterArgMax = secondLoopNest.loops.back().getRegionIterArg(1);
  Value iterArgSum = secondLoopNest.loops.back().getRegionIterArg(2);

  OpBuilder::InsertionGuard guardSecondLoop(rewriter);
  rewriter.setInsertionPointToStart(secondLoopNest.loops.back().getBody());

  // Extract slices
  auto [keySlice, valueSlice] =
      extractSlices(key, value, keyShape, ivs,
                    keyValueTileLength, headDimension, elementType, loc, rewriter);

  ret = bufferization::allocateTensorForShapedValue(
                rewriter, loc, keySlice, false, options, true);
  if (failed(ret)) {
    return {};
  }
  keySlice = ret.value();

  ret = bufferization::allocateTensorForShapedValue(
                rewriter, loc, valueSlice, false, options, true);
  if (failed(ret)) {
    return {};
  }
  valueSlice = ret.value();

  // Construct third loop
  int64_t tileSize{16};
  OpFoldResult warpSize = rewriter.getIndexAttr(tileSize);
  // Number of warps to distribute on
  OpFoldResult numWarps = rewriter.getIndexAttr(4);
  SmallVector<Attribute> warpMapping{mlir::gpu::GPUWarpMappingAttr::get(rewriter.getContext(), mlir::gpu::Warps::DimX)};
  scf::ForallOp forallOp = rewriter.create<scf::ForallOp>(
      loc, numWarps, ValueRange({iterArgResult, iterArgMax, iterArgSum}),
      rewriter.getArrayAttr(warpMapping));
  auto threadIds = llvm::to_vector(forallOp.getInductionVars());
  ivs.push_back(threadIds[0]);
  ops.push_back(forallOp);
  ArrayRef<BlockArgument> bbArgs = forallOp.getOutputBlockArguments();
  assert(bbArgs.size() == 3);
  Value iterArgResultInner = bbArgs[0];
  Value iterArgMaxInner = bbArgs[1];
  Value iterArgSumInner = bbArgs[2];

  OpBuilder::InsertionGuard guardThirdLoop(rewriter);
  rewriter.setInsertionPointToStart(forallOp.getBody(0));

  AffineExpr d0;
  bindDims(rewriter.getContext(), d0);
  auto threadMap =
      AffineMap::get(1, 0, {d0 * rewriter.getAffineConstantExpr(tileSize)},
                     rewriter.getContext());
  ivs[2] = rewriter.create<affine::AffineApplyOp>(loc, threadMap, ivs[2]);

  // Extract slices
  auto [querySlice, outputSlice, maxSlice, sumSlice] = extractSlicesInner(
      query, iterArgResultInner, iterArgMaxInner, iterArgSumInner, queryShape,
      tileSize, ivs, warpSize, headDimension, elementType, loc, rewriter);

  // Create body of innermost loop
  auto [result, newMax, newSum] = createAttentionBody(
      keySlice, valueSlice, querySlice, outputSlice, maxSlice, sumSlice, warpSize,
      sequenceTileLength, keyValueTileLength, headDimension, elementType, ops, loc, rewriter);

  OpBuilder::InsertionGuard guardThirdLoopEnd(rewriter);
  rewriter.setInsertionPointToStart(forallOp.getTerminator().getBody());

  // Insert slices inner
  insertSlicesInner(result, iterArgResultInner, newMax, iterArgMaxInner, newSum,
                    iterArgSumInner, queryShape, ivs, warpSize, headDimension, loc, rewriter);

  if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
          secondLoopNest.loops.back().getBody()->getTerminator())) {
    // Insert slices
    OpBuilder::InsertionGuard yieldGuard(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    result = forallOp.getResult(0);
    newMax = forallOp.getResult(1);
    newSum = forallOp.getResult(2);
    auto [updatedAcc, updatedMax, updatedSum] = insertSlices(
        result, iterArgResult, newMax, iterArgMax, newSum, iterArgSum, queryShape,
        ivs, sequenceTileLength, headDimension, loc, rewriter);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(
        yieldOp, ValueRange{updatedAcc, updatedMax, updatedSum});
  }

  if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
          firstLoopNest.loops.back().getBody()->getTerminator())) {
    OpBuilder::InsertionGuard yieldGuard(rewriter);
    rewriter.setInsertionPoint(yieldOp);

    secondLoopNest.results[0] = applyFinalScaling(secondLoopNest.results[0], secondLoopNest.results[2], loc, rewriter, ops);

    SmallVector<OpFoldResult> shape = tensor::createDimValues(rewriter, loc, secondLoopNest.results[0]);
    Value scratch = rewriter.create<tensor::EmptyOp>(loc, shape, rewriter.getF16Type());
    result = truncateToF16<2>(secondLoopNest.results[0], scratch, rewriter, loc, ops);
    auto one = rewriter.getIndexAttr(1);
    auto zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> strides(shape.size() + 1, one);
    SmallVector<OpFoldResult> sizes(shape.size() + 1, one);
    SmallVector<OpFoldResult> offsets(shape.size() + 1, zero);
    sizes[1] = sequenceTileLength;
    sizes[2] = headDimension;
    offsets[0] = ivs[0];
    result = rewriter.create<tensor::InsertSliceOp>(
        loc, result, iterArg, offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(
        yieldOp, ValueRange{result});
  }

  //OpBuilder::InsertionGuard forGuard(rewriter);
  //rewriter.setInsertionPointAfter(firstLoopNest.loops[0]);
  //ArrayRef<int64_t> resultShape = firstLoopNest.results[0].getType().cast<ShapedType>().getShape();
  //Value scratch = rewriter.create<tensor::EmptyOp>(loc, resultShape, rewriter.getF16Type());
  //result = truncateToF16<3>(firstLoopNest.results[0], output, rewriter, loc);

  attnOp->getParentOfType<ModuleOp>().dump();
  attnOp.getResults()[0].replaceAllUsesWith(firstLoopNest.results[0]);
  return ops;
}

namespace {

/// This is an implementation of flash attention which
/// is a tiled and fused implementation of the attention operator.
/// The attention operator computes:
/// matmul(softmax(matmul(Q, transpose(K))), V)
/// where: Q is the query matrix [B x N x d]
///        K is the key matrix   [B x N x d]
///        V is the value matrix [B x N x d]
///
/// The core algorithm is as follows:
/// For each element in B,
/// 1. Load a tile from the Q matrix of size T x d -> q
/// 2. Initialize statistics: running_sum, running_max
/// 3. for i = 0 to N with step T
///    a. Load a tile from the K matrix of size T x d -> k
///    a. Load a tile from the V matrix of size T x d -> v
///    b. Transpose(k) -> kT
///    c. Compute matmul(q, kT) -> qkT
///    d. Compute sum(qkT) along rows -> current_sum
///    e. Compute max(qkT) along rows -> current_max
///    f. Compute new max: max(current_max, running_max)
///    g. Compute new sum: alpha * running_sum + beta * current_sum
///    h. Compute curent estimate of softmax: exp(qKT - current_max) -> s
///    i. Scale softmax estimate and current value of result by
///       appropriate factors
///    j. Compute matmul(s, v) and add to accumulator
///
///
LogicalResult reifyAttentionTransform(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  funcOp.walk([&](IREE::LinalgExt::AttentionOp attnOp) {
    tileAndDecomposeAttention(attnOp, rewriter);
    return WalkResult::advance();
  });
  return success();
}

} // namespace

namespace {
struct TileAndDecomposeAttentionPass
    : public TileAndDecomposeAttentionBase<TileAndDecomposeAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void TileAndDecomposeAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  if (failed(reifyAttentionTransform(getOperation())))
    return signalPassFailure();
}

std::unique_ptr<Pass> createTileAndDecomposeAttentionPass() {
  return std::make_unique<TileAndDecomposeAttentionPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
