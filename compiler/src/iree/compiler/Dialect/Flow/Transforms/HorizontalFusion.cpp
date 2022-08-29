// Copyright 2021 Nod Labs

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Fuse N matmuls with common input =>
// matmul1 : M X K x N1
// matmul2:  M x K X N2
// ...
// matmulN:  M x K X NN
// combine to matmul3 : M x K x (N1 + N2 + ... + NN)
class HorizontallyFuseMatmulOp
    : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {

    //matmulOp->getParentOfType<ModuleOp>()->dump();
    auto loc = matmulOp.getLoc();
    Value input = matmulOp.getInputOperand(0)->get();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    // Input shape : M x K
    auto inputShape = inputType.getShape();
    auto elementTy = inputType.getElementType();

    SmallVector<linalg::MatmulOp> candidates;
    candidates.push_back(matmulOp);
    for (Operation *user : input.getUsers()) {
      if (auto candidateMatmulOp = dyn_cast<linalg::MatmulOp>(user)) {
        if (candidateMatmulOp != matmulOp)
          candidates.push_back(candidateMatmulOp);
      }
    }

    if (candidates.size() > 1) {
      SmallVector<Value, 2> filters;
      SmallVector<Value, 2> outputs;
      SmallVector<int64_t, 2> oldShapes;
      SmallVector<int64_t> filterShape{0, 0};
      SmallVector<int64_t> outputShape{0, 0};
      for (int i = 0; i < candidates.size(); i++) {
        // Output shape: M x N
        auto output = candidates[i].getOutputOperand(0)->get();
        auto outputShape = output.getType().cast<RankedTensorType>().getShape();
        // Filter shape: K x N
        filterShape[1] += outputShape[1];
        filterShape[0] = inputShape[1];
        filters.push_back(candidates[i].getInputOperand(1)->get());
        outputs.push_back(output);
        oldShapes.push_back(outputShape[1]);
      }

      // Create new tensor
      SmallVector<Value> dynDims;
      auto newFilter = rewriter.create<linalg::InitTensorOp>(loc, dynDims, filterShape, elementTy).getResult();
      auto newOutput = rewriter.create<linalg::InitTensorOp>(loc, dynDims, SmallVector<int64_t>{inputShape[0], filterShape[1]}, elementTy).getResult();

      // Fill output tensor with 0
      Value zeroVal = rewriter.createOrFold<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementTy));
      newOutput = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{newOutput}).result();

      // Insert slices
      int idx{0};
      for (int i = 0; i < filters.size(); i++) {
        newFilter = rewriter.create<tensor::InsertSliceOp>(loc, newFilter.getType(), filters[i], newFilter,
                                                           ValueRange({}), ValueRange({}), ValueRange({}),
                                                           rewriter.getI64ArrayAttr({0, idx}),
                                                           rewriter.getI64ArrayAttr({filterShape[0], oldShapes[i]}),
                                                           rewriter.getI64ArrayAttr({1, 1}));
        idx += oldShapes[i];
      }

      // Create new matmul
      auto outputType = RankedTensorType::get({inputShape[0], filterShape[1]}, elementTy);
      auto result = rewriter.create<linalg::MatmulOp>(
        loc, outputType, ArrayRef<Value>{input, newFilter}, ArrayRef<Value>{newOutput}
      ).getResult(0);

      //matmulOp->getParentOfType<ModuleOp>()->dump();
      // Extract slices
      SmallVector<Value> outputSlices;
      idx = 0;
      for (int i = 0; i < filters.size(); i++) {
        auto resultType = RankedTensorType::get({inputShape[0], oldShapes[i]}, elementTy);
        auto slice = rewriter.create<tensor::ExtractSliceOp>(loc, resultType, result, ValueRange({}), ValueRange({}), ValueRange({}),
                                                             rewriter.getI64ArrayAttr({0, idx}),
                                                             rewriter.getI64ArrayAttr({inputShape[0], oldShapes[i]}),
                                                             rewriter.getI64ArrayAttr({1, 1}));
        outputSlices.push_back(slice);
        idx += oldShapes[i];
      }

      //matmulOp->getParentOfType<ModuleOp>()->dump();
      // Replace results with new slices
      for (int i = 0; i < candidates.size(); i++) {
        rewriter.replaceOp(candidates[i], outputSlices[i]);
      }
      return success();
    }

    return failure();
  }
};

struct HorizontalFusionPass
    : public HorizontalFusionBase<HorizontalFusionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HorizontallyFuseMatmulOp
        >(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createHorizontalFusionPass() {
  return std::make_unique<HorizontalFusionPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
