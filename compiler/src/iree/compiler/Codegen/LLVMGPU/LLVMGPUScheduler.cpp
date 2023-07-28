// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <queue>

#define DEBUG_TYPE "iree-codegen-gpu-scheduler"

namespace mlir {
namespace iree_compiler {

namespace {

struct Node {
  Operation *op;
  SmallVector<Operation *> additionalNeighbors;
};

struct OperationComparator {
  bool operator()(Operation *lhs, Operation *rhs) {
    return costMap.at(lhs) < costMap.at(rhs);
  }
  DenseMap<Operation *, int> costMap;
};

bool isOpReady(Operation *op, DenseSet<Operation *> &unscheduledOps,
               DenseMap<Operation *, DenseSet<Operation *>> &additionalDependencies) {
  // Check additional dependencies
  if (additionalDependencies.contains(op)) {
    for (Operation *parent : additionalDependencies.at(op)) {
      if (unscheduledOps.contains(parent))
        return false;
    }
  }
  // An operation is ready to be scheduled if all its operands are ready. An
  // operation is ready if:
  const auto isReady = [&](Value value) {
    Operation *parent = value.getDefiningOp();
    // - it is a block argument,
    if (!parent)
      return true;
    // - or it is not defined by an unscheduled op (and also not nested within
    //   an unscheduled op).
    do {
      // Stop traversal when op under examination is reached.
      if (parent == op)
        return true;
      if (unscheduledOps.contains(parent))
        return false;
    } while ((parent = parent->getParentOp()));
    // No unscheduled op found.
    return true;
  };
  WalkResult readyToSchedule = op->walk([&](Operation *nestedOp) {
    return llvm::all_of(nestedOp->getOperands(),
                        [&](Value operand) { return isReady(operand); })
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return !readyToSchedule.wasInterrupted();
}

int delay(Operation *op) {
  if (auto loadOp = dyn_cast<nvgpu::LdMatrixOp>(op)) {
    return 1;
  }
  if (auto asyncCopyOp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op)) {
    return 1;
  }
  return 1;
}

bool prioritizedTopologicalSort(Block *block) {
  if (block->empty()) return true;

  // Keep track of additional edges (beyond what is in the IR) based on side effects
  llvm::iterator_range<Block::iterator> ops = block->without_terminator();
  Operation *lastOpWithMemoryEffects{nullptr};

  DenseMap<Operation *, DenseSet<Operation *>> additionalDependencies;
  auto updateEdges = [&](Operation *from, Operation *to) {
    if (!additionalDependencies.contains(to)) {
      additionalDependencies[to] = {};
    }
    additionalDependencies[to].insert(from);
    LLVM_DEBUG({
      llvm::dbgs() << "Creating an edge between : \n";
      llvm::dbgs() << "From : \n";
      from->dump();
      llvm::dbgs() << "To : \n";
      to->dump();
    });
  };

  // Assumptions: We start with an op with unknown memory effects.
  // In addition we assume all read/write effect ops between these
  // unknown memory effect ops are independent of each other and
  // only dependent on the unknown memory effect ops.
  DenseSet<Operation *> readWriteOps;
  for (Operation &op : ops) {
    if (isMemoryEffectFree(&op)) {
      if (readWriteOps.empty()) continue;
      for (Value operand : op.getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (readWriteOps.contains(producer)) {
          readWriteOps.erase(producer);
          readWriteOps.insert(&op);
        }
      }
      continue;
    }
    auto interface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!interface) {
      if (!lastOpWithMemoryEffects) {
        lastOpWithMemoryEffects = &op;
        continue;
      }
      if (readWriteOps.empty()) {
        updateEdges(lastOpWithMemoryEffects, &op);
      } else {
        for (Operation *readWriteOp : readWriteOps) {
          updateEdges(readWriteOp, &op);
        }
        readWriteOps.clear();
      }
      lastOpWithMemoryEffects = &op;
    } else {
      bool hasWriteEffect = interface.hasEffect<MemoryEffects::Write>();
      bool hasReadEffect = interface.hasEffect<MemoryEffects::Read>();
      if (lastOpWithMemoryEffects && (hasWriteEffect || hasReadEffect)) {
        updateEdges(lastOpWithMemoryEffects, &op);
        readWriteOps.insert(&op);
      }
    }
  }

  for (auto [key, value] : additionalDependencies) {
    LLVM_DEBUG({
      llvm::dbgs() << "Op = \n";
      key->dump();
      for (Operation *v : value) {
        llvm::dbgs() << "Value = \n";
        v->dump();
      }
    });
  }

  // Next, iterate block in reverse and assign costs to each node
  DenseMap<Operation *, int> costMap;
  std::queue<Operation *> unweightedOps;
  Operation *terminator = block->getTerminator();
  unweightedOps.push(terminator);
  costMap[terminator] = 0;
  while (!unweightedOps.empty()) {
    Operation *op = unweightedOps.front();
    unweightedOps.pop();
    for (Value operand : op->getOperands()) {
      if (Operation *producer = operand.getDefiningOp()) {
        if (costMap.contains(producer)) continue;
        costMap[producer] = costMap[op] + delay(op);
        LLVM_DEBUG({
          llvm::dbgs() << "Processing ...\n";
          producer->dump();
          llvm::dbgs() << "Operation has cost = " << costMap[producer] << "\n";
        });
        unweightedOps.push(producer);
      }
    }
    // Next, iterate over additional edges
    if (additionalDependencies.contains(op)) {
      for (Operation *producer : additionalDependencies.at(op)) {
        if (costMap.contains(producer)) continue;
        costMap[producer] = costMap[op] + 1;
        LLVM_DEBUG({
          llvm::dbgs() << "Processing ...\n";
          producer->dump();
          llvm::dbgs() << "Operation has cost = " << costMap[producer] << "\n";
        });
        unweightedOps.push(producer);
      }
    }
  }

  // Mark all operations as unscheduled.
  // Set cost for ops like dealloc.
  DenseSet<Operation *> unscheduledOps;
  ops = block->without_terminator();
  for (Operation &op : ops) {
    if (!costMap.contains(&op)) {
      LLVM_DEBUG({
        llvm::dbgs() << "Op does not have a cost! Setting to 0\n";
        op.dump();
      });
      costMap[&op] = -1;
    }
    unscheduledOps.insert(&op);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Total # of unscheduled ops = " << unscheduledOps.size() << "\n";
  });

  Block::iterator nextScheduledOp = ops.begin();
  Block::iterator end = ops.end();
  auto comparator = OperationComparator{costMap};

  bool allOpsScheduled = true;
  while (!unscheduledOps.empty()) {
    // Find list of ready ops and sort them based on their priority
    SmallVector<Operation *> readyOps;
    for (Operation &op :
         llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
      if (!isOpReady(&op, unscheduledOps, additionalDependencies)) {
        continue;
      }
      if (costMap[&op] < 0) {
        continue;
      }
      //LLVM_DEBUG({
      //  llvm::dbgs() << "Processing op = \n";
      //  op.dump();
      //});
      //LLVM_DEBUG(llvm::dbgs() << "Op is ready to be scheduled.\n");
      readyOps.push_back(&op);
    }

    auto sortedReadyOps = std::priority_queue<Operation *, std::vector<Operation *>, OperationComparator>{
      readyOps.begin(), readyOps.end(), comparator};

    LLVM_DEBUG({
      llvm::dbgs() << "# of ready ops: "<< sortedReadyOps.size() << "\n";
    });

    // Schedule the ready operations by moving them to the start.
    bool scheduledAtLeastOnce = false;
    LLVM_DEBUG(llvm::dbgs() << "==================\n");
    while (!sortedReadyOps.empty()) {
      Operation *op = sortedReadyOps.top();
      sortedReadyOps.pop();
      LLVM_DEBUG({
        llvm::dbgs() << "Scheduling op with cost = [" << costMap.at(op) << "] : \n";
        op->dump();
      });
      unscheduledOps.erase(op);
      op->moveBefore(block, nextScheduledOp);
      scheduledAtLeastOnce = true;
      // Move the iterator forward if we schedule the operation at the front.
      if (op == &*nextScheduledOp)
        ++nextScheduledOp;
    }

    // If no operations were scheduled, give up and advance the iterator.
    if (!scheduledAtLeastOnce) {
      allOpsScheduled = false;
      unscheduledOps.erase(&*nextScheduledOp);
      ++nextScheduledOp;
    }

  }

  return allOpsScheduled;
}

void scheduleOperations(func::FuncOp funcOp) {
  SmallVector<scf::ForOp> forOps;
  funcOp.walk([&](Operation *op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      forOps.push_back(forOp);
    }
    return WalkResult::advance();
  });

  // Only schedule body of inner-most for loop for now
  for (scf::ForOp forOp : forOps) {
    prioritizedTopologicalSort(&forOp.getLoopBody().front());
  }
}

struct LLVMGPUSchedulerPass
    : public LLVMGPUSchedulerBase<
          LLVMGPUSchedulerPass> {
  LLVMGPUSchedulerPass() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    scheduleOperations(getOperation());
  }

};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUSchedulerPass() {
  return std::make_unique<LLVMGPUSchedulerPass>();
}

} // namespace iree_compiler
} // namespace mlir
