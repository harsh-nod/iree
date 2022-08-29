// Copyright 2021 Nod Labs

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

class BringMatmulsTogether
    : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Value input = matmulOp.getInputOperand(0)->get();

    SmallVector<linalg::MatmulOp> candidates;
    candidates.push_back(matmulOp);
    for (Operation *user : input.getUsers()) {
      if (auto candidateMatmulOp = dyn_cast<linalg::MatmulOp>(user)) {
        if (candidateMatmulOp != matmulOp)
          candidates.push_back(candidateMatmulOp);
      }
    }

    if (candidates.size() > 1) {
      Operation *before;
      Operation *after;
      if (candidates[0]->isBeforeInBlock(candidates[1])) {
        before = candidates[0];
        after = candidates[1];
      } else {
        before = candidates[1];
        after = candidates[0];
      }
      rewriter.setInsertionPointAfter(after);

      // Now recursively move before's users that dominate after
      // below after
      BlockAndValueMapping bvm;
      std::vector<Operation *> visited;
      std::vector<Operation *> candidates;
      std::vector<Operation *> clones;
      std::vector<Operation *> toRemove;
      auto isNotVisited = [&visited] (Operation *user) {
        return std::find(visited.begin(), visited.end(), user) == visited.end();
      };
      candidates.push_back(before);
      visited.push_back(before);
      while (!candidates.empty()) {
        auto node = candidates[0];
        candidates.erase(candidates.begin());
        Operation *cloneOp{nullptr};
        if (node != before) {
          cloneOp = rewriter.clone(*node, bvm);
          clones.push_back(cloneOp);
          toRemove.push_back(node);
        }
        for (OpOperand &use : node->getUses()) {
          auto user = use.getOwner();
          if (user->isBeforeInBlock(after) && isNotVisited(user)) {
            candidates.push_back(user);
            visited.push_back(user);
            if (cloneOp) bvm.map(use.get(), cloneOp->getResult(0));
          }
        }
      }
      if (toRemove.empty()) return failure();
      for (int i = toRemove.size() - 1; i >= 0; i--) {
        toRemove[i]->replaceAllUsesWith(clones[i]);
        break;
      }
      return success();
    }
    return failure();
  }
};

struct BringMatmulsTogetherPass
    : public BringMatmulsTogetherBase<BringMatmulsTogetherPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<BringMatmulsTogether
        >(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createBringMatmulsTogetherPass() {
  return std::make_unique<BringMatmulsTogetherPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
