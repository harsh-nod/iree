// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "llvm/ADT/None.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"


namespace mlir {
namespace iree_compiler {

namespace {

LogicalResult hoistOutputs(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  funcOp.walk([&](gpu::SubgroupMmaStoreMatrixOp storeOp) {
    Value dst = storeOp.getDstMemref();
    auto dstType = dst.getType().cast<MemRefType>();
    ArrayRef<int64_t> dstShape = dstType.getShape();
    if (dstShape.size() != 2)
      return WalkResult::advance();
    if (dstShape[0] == dstShape[1])
      return WalkResult::advance();

    // Find promoted value from cast op
    auto castOp = dst.getDefiningOp<memref::CastOp>();
    if (!castOp)
      return WalkResult::advance();

    Value promotedOutput = castOp.getSource();
    // Find alloc that promoted the output
    auto allocOp = promotedOutput.getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return WalkResult::advance();

    // Find enclosing scf.for
    auto forOp = allocOp->getParentOfType<scf::ForOp>();
    if (!forOp)
      return WalkResult::advance();

    // Hoist alloc out of loop
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(forOp);

    printf("got one!\n");
    auto newAllocOp = rewriter.create<memref::AllocOp>(forOp.getLoc(),
                          allocOp.getResult().getType().cast<MemRefType>(),
                          rewriter.getIntegerAttr(rewriter.getIntegerType(64), *allocOp.getAlignment()));
    Value hoistedOutput = newAllocOp->getResult(0);
    allocOp.getResult().replaceAllUsesWith(hoistedOutput);

    // Hoist linalg generic(s) out of loop and erase vector.transfer_write
    SmallVector<Operation *> toErase, newCopies;
    Value output;
    for (OpOperand &use : hoistedOutput.getUses()) {
      auto linalgOp = dyn_cast<linalg::GenericOp>(use.getOwner());
      if (linalgOp) {
        if (linalgOp.isDpsInit(&use) || linalgOp.isDpsInput(&use)) {
          bool isInput = linalgOp.isDpsInput(&use);

          // Get the iterator types for the operand.
          SmallVector<utils::IteratorType> iteratorTypes = linalgOp.getIteratorTypesArray();
          // Get the indexing maps.
          auto indexingMaps = linalgOp.getIndexingMapsArray();
          // Get the input operands.
          SmallVector<Value> inputOperands;
          if (isInput) {
            inputOperands.push_back(hoistedOutput);
          } else {
            inputOperands.reserve(linalgOp.getNumDpsInputs());
            for (Value input : linalgOp.getInputs()) {
              inputOperands.push_back(input);
            }
            auto subviewOp = inputOperands[0].getDefiningOp<memref::SubViewOp>();
            if (!subviewOp) return WalkResult::advance();
            output = subviewOp.getSource();
          }

          // Get the output operands and result types.
          SmallVector<Type> resultTypes{};
          SmallVector<Value> outputOperands;
          if (isInput) {
            outputOperands.reserve(linalgOp.getNumDpsInits());
            for (Value output : linalgOp.getOutputs()) {
              outputOperands.push_back(output);
            }
          } else {
            outputOperands.push_back(hoistedOutput);
          }

          OpBuilder::InsertionGuard g2(rewriter);
          if (isInput) {
            rewriter.setInsertionPointAfter(forOp);
          } else {
            rewriter.setInsertionPoint(forOp);
          }
          auto newLinalgOp = rewriter.create<linalg::GenericOp>(forOp.getLoc(),
            resultTypes, inputOperands, outputOperands, indexingMaps,
            iteratorTypes, [](OpBuilder &builder, Location loc, ValueRange args) {
              builder.create<linalg::YieldOp>(loc, args[0]);
            });
          newCopies.push_back(newLinalgOp);
          toErase.push_back(linalgOp);
        }
      }
      auto deallocOp = dyn_cast<memref::DeallocOp>(use.getOwner());
      if (deallocOp) {
        toErase.push_back(deallocOp);
      }
    }

    OpBuilder::InsertionGuard g4(rewriter);
    rewriter.setInsertionPointAfter(newCopies.back());
    rewriter.create<memref::DeallocOp>(forOp.getLoc(), hoistedOutput);

    for (int i = 0; i < toErase.size(); i++) {
      rewriter.eraseOp(toErase[i]);
    }

    for (OpOperand &use : output.getUses()) {
      auto transferReadOp = dyn_cast<vector::TransferReadOp>(use.getOwner());
      if (transferReadOp) {
        OpBuilder::InsertionGuard g4(rewriter);
        rewriter.setInsertionPoint(transferReadOp);
        auto currentIndices = transferReadOp.getIndices();
        assert(currentIndices.size() == 3);
        // We need to get the gpu.thread_id from the affine map
        auto affineApplyOp = currentIndices[1].getDefiningOp<AffineApplyOp>();
        if (!affineApplyOp)
          return WalkResult::advance();
        auto affineOperands = affineApplyOp.getMapOperands();
        assert(affineOperands.size() == 2);
        SmallVector<Value> indices{affineOperands[1], currentIndices[2]};
        SmallVector<bool> inBounds{true};
        auto newReadOp = rewriter.create<vector::TransferReadOp>(transferReadOp.getLoc(),
            transferReadOp.getVector().getType(),
            hoistedOutput, indices,
            transferReadOp.getPadding(), inBounds);
        transferReadOp.getVector().replaceAllUsesWith(newReadOp.getVector());
        rewriter.eraseOp(transferReadOp);
        break;
      }
    }

    return WalkResult::advance();
  });
  return success();
}

} // namespace

namespace {
struct HoistOutputPass
    : public HoistOutputBase<HoistOutputPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    if (failed(hoistOutputs(getOperation())))
      return signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createHoistOutputPass() {
  return std::make_unique<HoistOutputPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
