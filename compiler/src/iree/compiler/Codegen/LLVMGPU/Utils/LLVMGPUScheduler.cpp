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
#include "llvm/ADT/iterator_range.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
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

bool isGraphBreak(Operation *op) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  return !interface;
}

bool hasReadWriteEffects(Operation *op) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) {
    return false;
  }
  bool hasWriteEffect = interface.hasEffect<MemoryEffects::Write>();
  bool hasReadEffect = interface.hasEffect<MemoryEffects::Read>();
  return hasWriteEffect || hasReadEffect;
}

// A node can consist of several ops.
// For example, we group:
//     %124 = vector.load %subview_2[%122, %123]
//     %125 = vector.insert_strided_slice %124, %cst_5
//     %168 = vector.extract %125[0, 0] : vector<4x4x4xf32>
//     as a single node since they essentially represent a "load" dependency.
struct Node {
  using Ptr = std::shared_ptr<Node>;
  SmallVector<Operation *> ops;
};

struct Graph {
  using Ptr = std::shared_ptr<Graph>;
  SetVector<Operation *> nodes;
  bool freeze{true};
};

static bool isOverwritten(Operation *op, ArrayAttr &offsets) {
  if (auto newInsertOp = dyn_cast<vector::InsertStridedSliceOp>(op)) {
    auto newOffsets = newInsertOp.getOffsets();
    return newOffsets == offsets;
  }
  return false;
}

static bool isExtracted(Operation *op, ArrayAttr offsets) {
  if (auto extractOp = dyn_cast<vector::ExtractOp>(op)) {
    auto newOffsets = extractOp.getPositionAttr();
    if (newOffsets.size() < offsets.size()) {
      bool match{true};
      for (int i = 0; i < newOffsets.size(); i++) {
        auto offset = dyn_cast<IntegerAttr>(offsets[i]);
        if (!offset) return false;
        match = match && (newOffsets[i] == offset.getInt());
      }
      return match;
    }
    return newOffsets == offsets;
  }
  return false;
}

static SmallVector<Operation *> createCompositeNode(Operation *op, DenseSet<Operation *> &ignore) {
  SmallVector<Operation *> groupedOps{op};
  if (isa<vector::LoadOp>(op)) {
    for (Operation *user : op->getUsers()) {
      if (auto insertOp = dyn_cast<vector::InsertStridedSliceOp>(user)) {
        groupedOps.push_back(insertOp);
        auto offsets = insertOp.getOffsets();

        // Find nodes at the end of the insert chain
        vector::InsertStridedSliceOp lastInsert{insertOp};
        do {
          SmallVector<Operation *> newUsers(insertOp->getUsers().begin(), insertOp->getUsers().end());
          if (newUsers.size() != 1) break;
          if (isOverwritten(newUsers[0], offsets)) break;
          lastInsert = insertOp;
          insertOp = dyn_cast_or_null<vector::InsertStridedSliceOp>(newUsers[0]);
        } while (insertOp);

        if (!insertOp)
          insertOp = lastInsert;

        // Find matching extract ops
        for (Operation *endUser : insertOp->getUsers()) {
          if (isExtracted(endUser, offsets)) {
            // Replace op with new op that uses the intermediate insert result
            auto extractOp = dyn_cast<vector::ExtractOp>(endUser);
            auto insertOp = dyn_cast<vector::InsertStridedSliceOp>(groupedOps.back());
            ignore.insert(extractOp);
            extractOp.setOperand(insertOp.getResult());
            groupedOps.push_back(extractOp);
          }
        }
      }
    }
  }
  return groupedOps;
}

static bool isComposite(Node::Ptr node) {
  return node->ops.size() > 1;
}

static bool isNodeReadyToSchedule(Node::Ptr node, SmallVector<Operation *> &unscheduledNodes,
                                  DenseMap<Operation *, Node::Ptr> &operatorToNode) {
  // Node is ready to schedule if all of its operands are ready.
  const auto isReady = [&](Value value) {
    Operation *parent = value.getDefiningOp();
    // If it is a block argument
    if (!parent) return true;
    // Or if it is not defined by an unscheduled op and not nested
    // within an unscheduled op
    do {
      if (std::find(unscheduledNodes.begin(), unscheduledNodes.end(), parent) != unscheduledNodes.end()) {
        LLVM_DEBUG({
          llvm::dbgs() << "Not scheduled because this op has not been scheduled yet ...\n";
          parent->dump();
        });
        return false;
      }
      if (operatorToNode[parent] == node) {
        return true;
      }
    } while ((parent = parent->getParentOp()));
    // No unscheduled op found
    return true;
  };

  // An operation is recursively ready to be scheduled of it and its nested
  // operations are ready.
  Operation *op = node->ops[0];
  WalkResult readyToSchedule = op->walk([&](Operation *nestedOp) {
    return llvm::all_of(nestedOp->getOperands(),
                        [&](Value operand) { return isReady(operand); })
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return !readyToSchedule.wasInterrupted();
}

struct NodeComparator {
  bool operator()(Node::Ptr lhs, Node::Ptr rhs) {
    return cost.at(lhs) < cost.at(rhs);
  }
  std::map<Node::Ptr, int> cost;
};

static bool prioritizedTopologicalSort(Block *block) {
  if (block->empty()) return true;
  llvm::iterator_range<Block::iterator> ops = block->without_terminator();
  if (ops.empty()) return true;

  // Create simple and composite nodes for ops
  DenseSet<Operation *> unscheduledOps, ignoreOps;
  DenseMap<Operation *, Node::Ptr> operationToNode;
  std::vector<Node::Ptr> nodes;

  struct Graph {
    SmallVector<Operation *> nodes;
    bool freeze{true};
  };
  SmallVector<Graph> unscheduledGraphs;
  unscheduledGraphs.push_back(Graph());

  for (Operation &op : ops) {
    LLVM_DEBUG({ llvm::dbgs() << "Processing ... \n"; });
    if (unscheduledOps.contains(&op)) {
      LLVM_DEBUG({
      llvm::dbgs() << "Already in unscheduled!\n";
      op.dump();
      });
      continue;
    }
    if (ignoreOps.contains(&op)) {
      LLVM_DEBUG({
      llvm::dbgs() << "in ignore list!\n";
      op.dump();
      });
      continue;
    }
    if (hasReadWriteEffects(&op)) {
      LLVM_DEBUG({
      llvm::dbgs() << "has read/write effects\n";
      op.dump();
      });
      auto compositeNode = std::make_shared<Node>();
      compositeNode->ops = createCompositeNode(&op, ignoreOps);
      for (Operation *child : compositeNode->ops) {
        operationToNode[child] = compositeNode;
        unscheduledOps.insert(child);
        unscheduledGraphs.back().nodes.push_back(child);
      }
      nodes.push_back(compositeNode);
      if (compositeNode->ops.size() > 1)
        unscheduledGraphs.back().freeze = false;
      continue;
    }
    if (isGraphBreak(&op)) {
      unscheduledGraphs.push_back(Graph());
    }

    auto newNode = std::make_shared<Node>();
    newNode->ops.push_back(&op);
    unscheduledOps.insert(&op);
    operationToNode[&op] = newNode;
    nodes.push_back(newNode);
    unscheduledGraphs.back().nodes.push_back(&op);
  }

  LLVM_DEBUG({
  llvm::dbgs() << "Partitioned graph into " << unscheduledGraphs.size() << " subgraphs \n";
  for (Graph graph : unscheduledGraphs) {
    if (graph.freeze) {
      llvm::dbgs() << "Graph is frozen\n";
    } else {
      llvm::dbgs() << "Graph is not frozen\n";
    }
  }
  });

  // Assign costs to nodes, starting with mfmas
  // The lower the cost, the higher the priority
  int count{0};
  int offset{10};
  int barrierCost{-1};
  std::map<Node::Ptr, int> nodeCost;
  std::map<Operation *, int> opCost;
  DenseSet<Operation *> prioritizedElemwiseOps;
  for (Node::Ptr node : nodes) {
    for (Operation *op : node->ops) {
      if (auto mfmaOp = dyn_cast<amdgpu::MFMAOp>(op)) {
        nodeCost[node] = count + offset;
        Operation *parentA = mfmaOp.getSourceA().getDefiningOp();
        Operation *parentB = mfmaOp.getSourceB().getDefiningOp();
        Operation *parentC = mfmaOp.getDestC().getDefiningOp();
        opCost[parentA] = opCost[parentB] = opCost[parentC] = nodeCost[node] - 1;
        count += offset;
        break;
      }
      if (auto barrierOp = dyn_cast<gpu::BarrierOp>(op)) {
        nodeCost[node] = barrierCost;
        break;
      }
      if (auto affineApplyOp = dyn_cast<affine::AffineApplyOp>(op)) {
        nodeCost[node] = 6;
        break;
      }
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        nodeCost[node] = 4;
        break;
      }
      if (auto loadGlobalOp = dyn_cast<vector::TransferReadOp>(op)) {
        nodeCost[node] = 7;
        break;
      }
      if (OpTrait::hasElementwiseMappableTraits(op)) {
        bool highPriority{true};
        for (auto operand : op->getOperands()) {
          Operation *constantOp = operand.getDefiningOp<arith::ConstantOp>();
          Operation *threadIdOp = operand.getDefiningOp<gpu::ThreadIdOp>();
          Operation *affineApplyOp = operand.getDefiningOp<affine::AffineApplyOp>();
          Operation *parent = operand.getDefiningOp();
          if (constantOp || threadIdOp || affineApplyOp || !parent || prioritizedElemwiseOps.contains(parent)) continue;
          highPriority = false;
          break;
        }
        if (highPriority) {
          prioritizedElemwiseOps.insert(op);
          nodeCost[node] = 5;
        }
      }
    }
  }

  int baselineValue{1000};
  for (Node::Ptr node : nodes) {
    if (nodeCost.count(node)) continue;
    for (Operation *op : node->ops) {
      if (opCost.count(op)) {
        nodeCost[node] = opCost[op];
        break;
      } else {
        nodeCost[node] = baselineValue;
      }
    }
  }

  Block::iterator nextScheduledOp = ops.begin();
  for (Graph graph : unscheduledGraphs) {
    if (graph.freeze) {
      continue;
    }
    while (!graph.nodes.empty()) {
        // Find the min cost node that can be scheduled
        Node::Ptr minCostNode{nullptr};
        for (Operation *op : graph.nodes) {
          Node::Ptr node = operationToNode[op];
          LLVM_DEBUG({
          llvm::dbgs() << "Attempting to schedule ... \n";
          op->dump();
          });
          if (!isNodeReadyToSchedule(node, graph.nodes, operationToNode)) {
            LLVM_DEBUG({
            llvm::dbgs() << "Op not ready to be scheduled \n";
            for (Operation *nodeOp : node->ops) {
              nodeOp->dump();
            }
            llvm::dbgs() << "----------\n";
            });
            continue;
          }
          if (!minCostNode) {
            minCostNode = node;
          }
          if (nodeCost[node] < nodeCost[minCostNode]) {
            minCostNode = node;
          }
        }

        LLVM_DEBUG({
        llvm::dbgs() << " nextScheduledOp = \n";
        nextScheduledOp->dump();
        });

        // Schedule the operation by moving it to the start
        for (Operation *nodeOp : minCostNode->ops) {
          LLVM_DEBUG({
          llvm::dbgs() << "Scheduling ...\n";
          nodeOp->dump();
          llvm::dbgs() << "----------\n";
          });
          nodeOp->moveBefore(block, nextScheduledOp);
          auto it = std::find(graph.nodes.begin(), graph.nodes.end(), nodeOp);
          if (it == graph.nodes.end())
            llvm::dbgs() << "Element not found!\n";
          graph.nodes.erase(it);
          if (nodeOp == &*nextScheduledOp)
            ++nextScheduledOp;
        }
        LLVM_DEBUG({
        llvm::dbgs() << "After ...\n";
        block->dump();
        llvm::dbgs() << "----------\n";
        });
    }
  }
  return true;
}

static bool insertBarriers(Block *block) {
  auto barrierOps = SmallVector<Operation *>(block->getOps<gpu::BarrierOp>());
  if (barrierOps.size() == 0) return true;
  IRRewriter rewriter(barrierOps[0]->getContext());
  // Erase all barriers inside
  for (auto barrierOp : barrierOps)
    rewriter.eraseOp(barrierOp);
  // Add barrier after compute
  auto mfmaOps = SmallVector<Operation *>(block->getOps<amdgpu::MFMAOp>());
  if (mfmaOps.size() == 0) return true;
  Location loc = mfmaOps.back()->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(mfmaOps.back());
  rewriter.create<gpu::BarrierOp>(loc);
  // Add barrier before global read
  auto globalLoads = SmallVector<Operation *>(block->getOps<vector::TransferReadOp>());
  if (globalLoads.size() == 0) return true;
  rewriter.setInsertionPointAfter(globalLoads.back());
  rewriter.create<gpu::BarrierOp>(loc);
  return true;
}

void scheduleOperations(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());

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
    insertBarriers(&forOp.getLoopBody().front());
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(forOp);
    rewriter.create<gpu::BarrierOp>(forOp.getLoc());
  }

  funcOp.walk([&](gpu::BarrierOp op) {
    if (op->hasAttr("__pipelining_first_stage__")) {
      rewriter.eraseOp(op);
    }
    return WalkResult::advance();
  });

}

} // namespace iree_compiler
} // namespace mlir