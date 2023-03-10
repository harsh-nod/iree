#include "iree/compiler/Codegen/Common/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/TilingUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include <iostream>

namespace mlir {
namespace iree_compiler {

namespace {

// 8-D Layout for a 1D or 2D vector
static constexpr int NDIM = 8;
namespace Dims {
  static constexpr int BATCH0 = 0;
  static constexpr int BATCH1 = 1;
  static constexpr int LANEIDZ = 2;
  static constexpr int LANEIDY = 3;
  static constexpr int LANEIDX = 4;
  static constexpr int VECIDZ = 5;
  static constexpr int VECIDY = 6;
  static constexpr int VECIDX = 7;
}

struct Layout {
  int64_t shape[NDIM];
  // Original order of indices
  int64_t rowOrder[4];
  int64_t colOrder[4];
  // Rank of tensor
  int64_t rank{1};
  // Indices for each dim (only for rank-2 vectors)
  int64_t indices[NDIM];

  std::tuple<int, int, int> unflatten(int laneId, int xMax, int yMax) {
    int laneIdz = laneId / (xMax * yMax);
    laneId -= (laneIdz * xMax * yMax);
    int laneIdy = laneId / xMax;
    int laneIdx = laneId % xMax;
    return std::make_tuple(laneIdx, laneIdy, laneIdz);
  }

  int flatten(int x, int y, int z, int xMax, int yMax) {
    return ((z * xMax * yMax) + (y * xMax) + x);
  }

  int computeDim(int64_t (&rowOrder)[4], int (&current)[NDIM]) {
    int row = 0;
    int rowScale = 1.0;
    for (int i = 0; i < 3; i++) {
      switch(rowOrder[i]) {
        case Dims::LANEIDX:
            row += rowScale * current[Dims::LANEIDX];
            rowScale *= shape[Dims::LANEIDX];
            break;
        case Dims::LANEIDY:
            row += rowScale * current[Dims::LANEIDY];
            rowScale *= shape[Dims::LANEIDY];
            break;
        case Dims::LANEIDZ:
            row += rowScale * current[Dims::LANEIDZ];
            rowScale *= shape[Dims::LANEIDZ];
            break;
        case Dims::VECIDX:
            row += rowScale * current[Dims::VECIDX];
            rowScale *= shape[Dims::VECIDX];
            break;
        case Dims::VECIDY:
            row += rowScale * current[Dims::VECIDY];
            rowScale *= shape[Dims::VECIDY];
            break;
        case Dims::VECIDZ:
            row += rowScale * current[Dims::VECIDZ];
            rowScale *= shape[Dims::VECIDZ];
            break;
      }
    }
    return row;
  }

  AffineExpr computeAffineExpr(int64_t (&rowOrder)[4], int batch0, int batch1, int vecIdZ, int vecIdY, int vecIdX, OpBuilder &builder) {
    AffineExpr d0, d1, d2;
    bindDims(builder.getContext(), d0, d1, d2);
    AffineExpr row = builder.getAffineConstantExpr(0);
    AffineExpr rowScale = builder.getAffineConstantExpr(1.0);
    for (int i = 0; i < 4; i++) {
      switch(rowOrder[i]) {
        case Dims::LANEIDX:
            row = row + rowScale * d0;
            rowScale = rowScale * builder.getAffineConstantExpr(shape[Dims::LANEIDX]);
            break;
        case Dims::LANEIDY:
            row = row + rowScale * d1;
            rowScale = rowScale * builder.getAffineConstantExpr(shape[Dims::LANEIDY]);
            break;
        case Dims::LANEIDZ:
            row = row + rowScale * d2;
            rowScale = rowScale * builder.getAffineConstantExpr(shape[Dims::LANEIDZ]);
            break;
        case Dims::VECIDX:
            row = row + rowScale * builder.getAffineConstantExpr(vecIdX);
            rowScale = rowScale * builder.getAffineConstantExpr(shape[Dims::VECIDX]);
            break;
        case Dims::VECIDY:
            row = row + rowScale * builder.getAffineConstantExpr(vecIdY);
            rowScale = rowScale * builder.getAffineConstantExpr(shape[Dims::VECIDY]);
            break;
        case Dims::VECIDZ:
            row = row + rowScale * builder.getAffineConstantExpr(vecIdZ);
            rowScale = rowScale * builder.getAffineConstantExpr(shape[Dims::VECIDZ]);
            break;
        case Dims::BATCH0:
            row = row + rowScale * builder.getAffineConstantExpr(batch0);
            rowScale = rowScale * builder.getAffineConstantExpr(shape[Dims::BATCH0]);
            break;
        case Dims::BATCH1:
            row = row + rowScale * builder.getAffineConstantExpr(batch1);
            rowScale = rowScale * builder.getAffineConstantExpr(shape[Dims::BATCH1]);
            break;
      }
    }
    return row;
  }

  // Ignoring batch dims for now
  // Compute the (row, col) for a given lane[i, j, k] and vec[a, b, c]
  std::tuple<int, int> computeRowCol(int laneId, int vecId) {
    auto [laneIdx, laneIdy, laneIdz] = unflatten(laneId, shape[Dims::LANEIDX], shape[Dims::LANEIDY]);
    auto [vecIdx, vecIdy, vecIdz] = unflatten(vecId, shape[Dims::VECIDX], shape[Dims::VECIDY]);
    int current[8] = {0, 0, laneIdz, laneIdy, laneIdx, vecIdz, vecIdy, vecIdx};
    int row = computeDim(rowOrder, current);
    int col = computeDim(colOrder, current);
    return std::make_tuple(row, col);
  }

  void printThreadMap(int numThreads, int numVec) {
    int m = 1;
    int n = 1;
    for (int i = 0; i < 3; i++) {
      m *= shape[rowOrder[i]];
      n *= shape[colOrder[i]];
    }
    int *dict = new int[m * n];
    for (int i = 0; i < numThreads; i++) {
      for (int j = 0; j < numVec; j++) {
        auto [row, col] = computeRowCol(i, j);
        dict[row * n + col] = i;
      }
    }
    for (int i = 0; i < m; i++) {
      std::stringstream ss;
      for (int j = 0; j < n; j++) {
        ss << dict[i *n + j] << " , ";
      }
      ss << "\n";
      std::cout << ss.str();
    }
    delete [] dict;
  }

  void setBatchId(const std::vector<int64_t> &batchId) {
    assert(batchId.size() == 2);
    shape[Dims::BATCH0] = batchId[0];
    shape[Dims::BATCH1] = batchId[1];
  }

  void setLaneId(const std::vector<int64_t> &laneId) {
    assert(laneId.size() == 3);
    shape[Dims::LANEIDZ] = laneId[0];
    shape[Dims::LANEIDY] = laneId[1];
    shape[Dims::LANEIDX] = laneId[2];
  }

  void setVecId(const std::vector<int64_t> &vecId) {
    assert(vecId.size() == 3);
    shape[Dims::VECIDZ] = vecId[0];
    shape[Dims::VECIDY] = vecId[1];
    shape[Dims::VECIDX] = vecId[2];
  }

  Layout(const Layout &copy) {
    memcpy(shape, copy.shape, sizeof(shape));
    memcpy(rowOrder, copy.rowOrder, sizeof(rowOrder));
    memcpy(colOrder, copy.colOrder, sizeof(colOrder));
    memcpy(indices, copy.indices, sizeof(indices));
    rank = copy.rank;
  }

  Layout(std::vector<int64_t> batchId, std::vector<int64_t> laneId, std::vector<int64_t> vecId, std::vector<int64_t> orderRows) {
    setBatchId(batchId);
    setLaneId(laneId);
    setVecId(vecId);
    rank = 1;
    assert(orderRows.size() == 3);
    for (int i = 0; i < 3; i++) {
      rowOrder[i] = orderRows[i];
      indices[rowOrder[i]] = 0;
    }
    rowOrder[3] = Dims::BATCH0;
    indices[Dims::BATCH0] = 0;
    indices[Dims::BATCH1] = 1;
  }

  Layout(std::vector<int64_t> batchId, std::vector<int64_t> laneId, std::vector<int64_t> vecId, std::vector<int64_t> orderRows, std::vector<int64_t> orderCols) {
    setBatchId(batchId);
    setLaneId(laneId);
    setVecId(vecId);
    rank = 2;
    assert(orderRows.size() == 3);
    for (int i = 0; i < 3; i++) {
      rowOrder[i] = orderRows[i];
      indices[rowOrder[i]] = 0;
    }
    rowOrder[3] = Dims::BATCH0;
    assert(orderCols.size() == 3);
    for (int i = 0; i < 3; i++) {
      colOrder[i] = orderCols[i];
      indices[colOrder[i]] = 1;
    }
    colOrder[3] = Dims::BATCH1;
    indices[Dims::BATCH0] = 0;
    indices[Dims::BATCH1] = 1;
  }

  void reduce(int dim) {
    for (int i = 0; i < NDIM; i++) {
      if (indices[i] == dim)
        shape[i] = 1;
    }
    rank = 1;
  }

  bool operator ==(const Layout &layout) const {
    for (int i = 0; i < NDIM; i++) {
      if (shape[i] != layout.shape[i])
        return false;
    }
    return true;
  }

  bool operator !=(const Layout &layout) const {
    for (int i = 0; i < NDIM; i++) {
      if (shape[i] != layout.shape[i])
        return true;
    }
    return false;
  }

  std::string str() const {
    std::stringstream layout;
    for (int i = 0; i < NDIM; i++) {
      layout << shape[i];
      if (i != NDIM - 1)
        layout << " x ";
    }
    return layout.str();
  }

  std::string indexStr() const {
    std::stringstream layout;
    for (int i = 0; i < NDIM; i++) {
      if (rank == 1)
        layout << "0";
      else
        layout << indices[i];
      if (i != NDIM - 1)
        layout << " x ";
    }
    return layout.str();
  }
};

void printLayout(std::string str, Value lhs, DenseMap<Value, Layout> &layoutMap) {
  if (layoutMap.count(lhs)) {
    Layout layout = layoutMap.at(lhs);
    std::cout << "Layout for " << str << " = " << layout.str() << " | " << layout.indexStr() << std::endl;
  }
}

void printLayoutMismatch(std::string str, const Layout &oldLayout, const Layout &newLayout) {
    printf("Layout transformation required!\n");
    std::cout << "Old layout = " << oldLayout.str() << std::endl;
    std::cout << "New layout = " << newLayout.str() << std::endl;
}

void handleMmaOperands(Value lhs, Layout &layout, DenseMap<Value, Layout> &layoutMap, std::string name) {
  if (layoutMap.count(lhs)) {
    if (layout != layoutMap.at(lhs)) {
      printLayoutMismatch("lhs", layoutMap.at(lhs), layout);
    }
  }
  layoutMap.try_emplace(lhs, layout);
  printLayout(name, lhs, layoutMap);
  if (auto readOp = lhs.getDefiningOp<vector::TransferReadOp>()) {
    Value src = readOp.getSource();
    layoutMap.try_emplace(src, layout);
    printLayout(name + " transfer_read ", lhs, layoutMap);
  }
}

// Initialize layouts for contract op (these follow from mma.sync.m16n8k16)
void setMmaContractLayouts(Value lhs, Value rhs, Value result, DenseMap<Value, Layout> &layoutMap) {
  ArrayRef<int64_t> lhsShape = lhs.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape = rhs.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> resultShape = result.getType().cast<ShapedType>().getShape();
  Layout lhsLayout({lhsShape[0] / 16, lhsShape[1] / 16}, {1, 8, 4}, {2, 2, 2},
                   {Dims::LANEIDY, Dims::VECIDZ, Dims::LANEIDZ}, {Dims::VECIDX, Dims::LANEIDX, Dims::VECIDY});
  //lhsLayout.printThreadMap(32, 8);
  handleMmaOperands(lhs, lhsLayout, layoutMap, "lhs");
  int mShape{16}, nShape{8};
  // Check if this is a transposed mma
  if (rhsShape[1] == lhsShape[1]) {
    std::swap(mShape, nShape);
  }
  Layout rhsLayout({rhsShape[0] / mShape, rhsShape[1] / nShape}, {1, 8, 4}, {1, 2, 2},
                    {Dims::VECIDX, Dims::LANEIDX, Dims::VECIDY}, {Dims::LANEIDY, Dims::LANEIDZ, Dims::VECIDZ});
  //rhsLayout.printThreadMap(32, 4);
  handleMmaOperands(rhs, rhsLayout, layoutMap, "rhs");

  Layout resultLayout({resultShape[0] / 16, resultShape[1] / 8}, {1, 8, 4}, {1, 2, 2},
                      {Dims::LANEIDY, Dims::VECIDY, Dims::LANEIDZ}, {Dims::VECIDX, Dims::LANEIDX, Dims::VECIDZ});
  //resultLayout.printThreadMap(32, 4);
  layoutMap.try_emplace(result, resultLayout);
  printLayout("result", result, layoutMap);
}

template <typename T>
void propagateFloatBinaryArithOps(Operation *op, DenseMap<Value, Layout> &layoutMap) {
  if (auto subOp = dyn_cast<T>(op)) {
    std::string name{"base"};
    if constexpr (std::is_same_v<T, arith::SubFOp>) {
      name = "subtracted";
    }
    if constexpr (std::is_same_v<T, arith::MulFOp>) {
      name = "multiplied";
    }
    if constexpr (std::is_same_v<T, arith::DivFOp>) {
      name = "divided";
    }
    Value lhs = subOp.getLhs();
    Value rhs = subOp.getRhs();
    Value result = subOp.getResult();
    if (layoutMap.count(lhs) && layoutMap.count(rhs)) {
      Layout lhsLayout = layoutMap.at(lhs);
      Layout rhsLayout = layoutMap.at(rhs);
      if (lhsLayout == rhsLayout) {
        layoutMap.try_emplace(result, lhsLayout);
        printLayout(name, result, layoutMap);
      }
    }
    if (layoutMap.count(lhs) && !layoutMap.count(rhs)) {
      Layout lhsLayout = layoutMap.at(lhs);
      layoutMap.try_emplace(rhs, lhsLayout);
      printLayout("rhs propagated", rhs, layoutMap);
      layoutMap.try_emplace(result, lhsLayout);
      printLayout(name, result, layoutMap);
    }
    if (layoutMap.count(rhs) && !layoutMap.count(lhs)) {
      Layout rhsLayout = layoutMap.at(rhs);
      layoutMap.try_emplace(lhs, rhsLayout);
      printLayout("lhs propagated", lhs, layoutMap);
      layoutMap.try_emplace(result, rhsLayout);
      printLayout(name, result, layoutMap);
    }
  }
}

void propagateLayout(Operation *op, DenseMap<Value, Layout> &layoutMap) {
  if (auto reductionOp = dyn_cast<vector::MultiDimReductionOp>(op)) {
    Value src = reductionOp.getSource();
    if (layoutMap.count(src)) {
      // Determine layout of result (after reduction)
      auto reductionDims = reductionOp.getReductionDims().getAsRange<IntegerAttr>();
      Value result = reductionOp.getResult();
      Layout resultLayout(layoutMap.at(src));
      for (auto ia : reductionDims) {
        resultLayout.reduce(ia.getInt());
      }
      layoutMap.try_emplace(result, resultLayout);
      printLayout("reduced", result, layoutMap);
      // Propagate result layout to accumulator
      Value acc = reductionOp.getAcc();
      if (layoutMap.count(acc)) {
        if (layoutMap.at(acc) != resultLayout) {
            printLayoutMismatch("accumulator", layoutMap.at(acc), resultLayout);
        }
      }
      layoutMap.try_emplace(acc, resultLayout);
      printLayout("accumulator", result, layoutMap);
    }
  }

  if (auto broadcastOp = dyn_cast<vector::BroadcastOp>(op)) {
    // For broadcast op, broadcast to shape of vector prior to reduction
    Value src = broadcastOp.getSource();
    if (auto reductionOp = src.getDefiningOp<vector::MultiDimReductionOp>()) {
      Value reductionSrc = reductionOp.getSource();
      Value result = broadcastOp.getResult();
      ArrayRef<int64_t> resultShape = result.getType().cast<ShapedType>().getShape();
      ArrayRef<int64_t> srcShape = reductionSrc.getType().cast<ShapedType>().getShape();
      assert(srcShape.size() == 2);
      SmallVector<int64_t> transposedSrcShape{srcShape[1], srcShape[0]};
      if ((srcShape != resultShape) && (transposedSrcShape != resultShape)) {
        // This is very specific to FA, will need to generalize
        Layout bLayout(layoutMap.at(reductionSrc));
        // Since we are going from 32x128 -> 32x64, divide batch by 2
        bLayout.shape[1] /= 2;
        layoutMap.try_emplace(result, bLayout);
        printLayout("broadcasted [special]", result, layoutMap);
        return;
      }
      if (layoutMap.count(reductionSrc))
        layoutMap.try_emplace(result, layoutMap.at(reductionSrc));
      printLayout("broadcasted", result, layoutMap);
    }
  }

  if (auto transposeOp = dyn_cast<vector::TransposeOp>(op)) {
    Value src = transposeOp.getVector();
    Value result = transposeOp.getResult();
    if (layoutMap.count(src)) {
      layoutMap.try_emplace(result, layoutMap.at(src));
      printLayout("transposed", result, layoutMap);
    }
  }

  propagateFloatBinaryArithOps<arith::SubFOp>(op, layoutMap);
  propagateFloatBinaryArithOps<arith::MulFOp>(op, layoutMap);
  propagateFloatBinaryArithOps<arith::DivFOp>(op, layoutMap);

  if (auto expOp = dyn_cast<math::ExpOp>(op)) {
    Value source = expOp.getOperand();
    if (layoutMap.count(source)) {
      Value result = expOp.getResult();
      layoutMap.try_emplace(result, layoutMap.at(source));
      printLayout("exponentiated", result, layoutMap);
    }
  }
}

void convertToSIMT(Operation *op, DenseMap<Value, Layout> &layoutMap, DenseMap<Value, Value> &simdToSimt, OpBuilder &builder) {
  Location loc = op->getLoc();
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    // Convert reads to memref.load
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(readOp);
    Value source = readOp.getSource();
    SmallVector<Value> indices = readOp.getIndices();
    Type elementType = source.getType().cast<ShapedType>().getElementType();
    Value threadIdX = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value threadIdY = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
    Value threadIdZ = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);
    Value result = readOp.getResult();
    Layout layout = layoutMap.at(result);
    auto vecType =
    VectorType::get(
        {layout.shape[Dims::BATCH0], layout.shape[Dims::BATCH1],
         layout.shape[Dims::VECIDZ] * layout.shape[Dims::VECIDY], layout.shape[Dims::VECIDX]},
         elementType);
    Value vector = builder.create<arith::ConstantOp>(loc, vecType, builder.getZeroAttr(vecType));
    for (int b0 = 0; b0 < layout.shape[Dims::BATCH0]; b0++) {
      for (int b1 = 0; b1 < layout.shape[Dims::BATCH1]; b1++) {
        for (int i = 0; i < layout.shape[Dims::VECIDZ]; i++) {
          for (int j = 0; j < layout.shape[Dims::VECIDY]; j++) {
            for (int k = 0; k < layout.shape[Dims::VECIDX]; k++) {
                AffineExpr row = layout.computeAffineExpr(layout.rowOrder, b0, b1, i, j, k, builder);
                AffineMap rowMap = AffineMap::get(3, 0, row, builder.getContext());
                Value rowIndex = builder.create<AffineApplyOp>(loc, rowMap, SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
                AffineExpr col = layout.computeAffineExpr(layout.colOrder, b0, b1, i, j, k, builder);
                AffineMap colMap = AffineMap::get(3, 0, col, builder.getContext());
                Value colIndex = builder.create<AffineApplyOp>(loc, colMap, SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
                if (layout.rank == 1)
                    indices.back() = rowIndex;
                if (layout.rank == 2) {
                    assert(indices.size() >= 2);
                    indices[indices.size() - 2] = rowIndex;
                    indices[indices.size() - 1] = colIndex;
                }
                Value el = builder.create<memref::LoadOp>(loc, source, indices);
                auto vectorType = VectorType::get({1}, elementType);
                Value v = builder.create<vector::BroadcastOp>(loc, vectorType, el);
                SmallVector<int64_t> offsets{b0, b1, i * layout.shape[Dims::VECIDY] + j, k};
                SmallVector<int64_t> strides{1};
                vector = builder.create<vector::InsertStridedSliceOp>(loc, v, vector, offsets, strides);
            }
          }
        }
      }
    }
    simdToSimt.try_emplace(result, vector);
  }

  if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(contractOp);
    Value lhs = contractOp.getLhs();
    if (!simdToSimt.count(lhs)) return;
    Type elementType = lhs.getType().cast<ShapedType>().getElementType();
    Value rhs = contractOp.getRhs();
    if (!simdToSimt.count(rhs)) return;
    Value contractResult = contractOp.getResult();
    Layout lhsLayout = layoutMap.at(lhs);
    Layout rhsLayout = layoutMap.at(rhs);
    Layout resultLayout = layoutMap.at(contractResult);
    SmallVector<int64_t> vecShape{resultLayout.shape[Dims::BATCH0], resultLayout.shape[Dims::BATCH1],
                                  resultLayout.shape[Dims::VECIDZ] * resultLayout.shape[Dims::VECIDY], resultLayout.shape[Dims::VECIDX]};
    auto vecType = VectorType::get(vecShape, elementType);
    Value result = builder.create<arith::ConstantOp>(loc, vecType, builder.getZeroAttr(vecType));
    int M = resultLayout.shape[Dims::BATCH0];
    int N = resultLayout.shape[Dims::BATCH1];
    int K = lhsLayout.shape[Dims::BATCH1];
    ArrayRef<int64_t> lhsShape = lhs.getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> rhsShape = rhs.getType().cast<ShapedType>().getShape();
    bool transpose = lhsShape[0] != rhsShape[1];
    auto cType = VectorType::get({vecShape[2], vecShape[3]}, elementType);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        Value cMatrix = builder.create<arith::ConstantOp>(loc, cType, builder.getZeroAttr(cType));
        for (int k = 0; k < K; k++) {
            Value aMatrix = builder.create<vector::ExtractOp>(loc,
                simdToSimt.at(lhs), SmallVector<int64_t>{i, k});
            SmallVector<int64_t> indices{k, j};
            if (transpose) indices = {j, k};
            Value bMatrix = builder.create<vector::ExtractOp>(loc,
                simdToSimt.at(rhs), indices);
            cMatrix = builder.create<nvgpu::MmaSyncOp>(loc, aMatrix, bMatrix, cMatrix, builder.getI64ArrayAttr({16, 8, 16}));
        }
        result = builder.create<vector::InsertOp>(loc, cMatrix, result, SmallVector<int64_t>{i, j});
      }
    }
    simdToSimt.try_emplace(contractResult, result);
  }

  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    // Convert writes to memref.store
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(writeOp);
    Value vector = writeOp.getVector();
    Value source = writeOp.getSource();
    SmallVector<Value> indices = writeOp.getIndices();
    Value threadIdX = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value threadIdY = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
    Value threadIdZ = builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);
    Value result = writeOp.getResult();
    if (!layoutMap.count(vector)) return;
    Layout layout = layoutMap.at(vector);
    for (int b0 = 0; b0 < layout.shape[Dims::BATCH0]; b0++) {
      for (int b1 = 0; b1 < layout.shape[Dims::BATCH1]; b1++) {
        for (int i = 0; i < layout.shape[Dims::VECIDZ]; i++) {
          for (int j = 0; j < layout.shape[Dims::VECIDY]; j++) {
            for (int k = 0; k < layout.shape[Dims::VECIDX]; k++) {
                Value v = builder.create<vector::ExtractOp>(loc, simdToSimt.at(vector), SmallVector<int64_t>{b0, b1, i * layout.shape[Dims::VECIDY] + j, k});
                AffineExpr row = layout.computeAffineExpr(layout.rowOrder, b0, b1, i, j, k, builder);
                AffineMap rowMap = AffineMap::get(3, 0, row, builder.getContext());
                Value rowIndex = builder.create<AffineApplyOp>(loc, rowMap, SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
                AffineExpr col = layout.computeAffineExpr(layout.colOrder, b0, b1, i, j, k, builder);
                AffineMap colMap = AffineMap::get(3, 0, col, builder.getContext());
                Value colIndex = builder.create<AffineApplyOp>(loc, colMap, SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
                if (layout.rank == 1)
                    indices.back() = rowIndex;
                if (layout.rank == 2) {
                    assert(indices.size() >= 2);
                    indices[indices.size() - 2] = rowIndex;
                    indices[indices.size() - 1] = colIndex;
                }
                builder.create<memref::StoreOp>(loc, v, source, indices);
            }
          }
        }
      }
    }
    simdToSimt.try_emplace(result, vector);
  }

  //op->getParentOfType<ModuleOp>().dump();
}

struct LayoutAnalysisAndDistributionPass
    : public LayoutAnalysisAndDistributionBase<LayoutAnalysisAndDistributionPass> {
 public:
  LayoutAnalysisAndDistributionPass() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, nvgpu::NVGPUDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    auto funcOp = getOperation();

    // First compute the layouts
    DenseMap<Value, Layout> layoutMap;
    funcOp.walk([&](Operation *op) {
      if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
        Value lhs = contractOp.getLhs();
        Value rhs = contractOp.getRhs();
        Value result = contractOp.getResult();
        setMmaContractLayouts(lhs, rhs, result, layoutMap);
      } else {
        propagateLayout(op, layoutMap);
      }
      return WalkResult::advance();
    });

    // Next, emit the SIMT code
    DenseMap<Value, Value> simdToSimt;
    funcOp.walk([&](Operation *op) {
      convertToSIMT(op, layoutMap, simdToSimt, rewriter);
      return WalkResult::advance();
    });

    // Erase all the old ops
    funcOp.walk([&](vector::TransferWriteOp op) {
      rewriter.eraseOp(op);
      return WalkResult::advance();
    });
    funcOp.walk([&](vector::ContractionOp op) {
      rewriter.eraseOp(op);
      return WalkResult::advance();
    });
    funcOp.walk([&](vector::TransferReadOp op) {
      rewriter.eraseOp(op);
      return WalkResult::advance();
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLayoutAnalysisAndDistributionPass() {
  return std::make_unique<LayoutAnalysisAndDistributionPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
