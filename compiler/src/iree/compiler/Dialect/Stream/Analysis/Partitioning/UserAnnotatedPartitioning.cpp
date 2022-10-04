// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-user-annotated-stream-partitioning"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

PartitionSet partitionStreamableOpsUserAnnotated(
    IREE::Stream::PartitioningConfigAttr config, Block *block) {
  PartitionSet partitionSet;

  struct PartitionBuilder {
    unsigned ordinal;
    // TODO: Unify this with affinity
    // Device of partition
    int device;
    // Affinity of the partition.
    IREE::Stream::AffinityAttr affinity;
    // Ops present in the partition; ops may be present in multiple partitions.
    SetVector<Operation *> ops;
  };
  SmallVector<std::unique_ptr<PartitionBuilder>> builders;
  llvm::BitVector usableBuilders;

  struct OpInfo {
    // Which partitions the op is contained within.
    llvm::BitVector membership;
    // Which partitions transitively depend on this operation.
    llvm::BitVector hazards;
  };
  DenseMap<Operation *, OpInfo> opInfos;

  for (auto &op : llvm::reverse(*block)) {
    // Skip constants; they just add noise (and since they are heavily CSE'd
    // they have lots of users to test).
    if (op.hasTrait<OpTrait::ConstantLike>()) {
      LLVM_DEBUG(llvm::dbgs() << "(ignoring constant)\n");
      continue;
    } else if (!isa<IREE::Stream::StreamableOpInterface>(op)) {
      // Not a streamable op. If it has side-effects then we force a hazard on
      // all builders so that we don't move ops across it.
      if (!mlir::wouldOpBeTriviallyDead(&op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "Side-effecting op forcing flush and freeze:\n";
          op.dump();
        });
        usableBuilders.reset();
      }
      // Even though not a streamable op we still want to track it below.
    }

    // Initialize op info for this op - whether streamable or not. We track
    // transitive hazards on each op. Note that thanks to the ordering of ops
    // in SSA form (_reversed here!_) we know that once we visit this op no
    // partition created after it can ever depend on it if it doesn't here. This
    // lets us keep the bitvectors small.
    auto &opInfo = opInfos[&op];
    opInfo.hazards.reserve(builders.size() + 1);
    opInfo.hazards.resize(builders.size(), /*t=*/false);

    IREE::Stream::AffinityAttr affinityAttr;
    if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
      affinityAttr = affinityOp.getAffinity();
    }

    // TODO: Integrate device with affinities
    int device{-1};
    if (op.hasAttr("device")) {
      device = op.getAttr("device").cast<IntegerAttr>().getValue().getSExtValue();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "====\nPartitioning op:\n";
      op.dump();
    });

    // Set bits for each partition this op may be able to be placed into.
    // We prune the set based on whether the users are part of a transitive
    // dependency chain down the use-def chain to a partition.
    llvm::BitVector consumers(builders.size(), /*t=*/false);
    for (auto user : op.getUsers()) {
      auto &userInfo = opInfos[user];
      LLVM_DEBUG({
        llvm::dbgs() << "Testing user:\n";
        user->dump();
        for (auto membershipOrdinal : userInfo.membership.set_bits()) {
          llvm::dbgs() << "  member of partition " << membershipOrdinal << "\n";
        }
        for (auto hazardOrdinal : userInfo.hazards.set_bits()) {
          llvm::dbgs() << "  hazard w/ partition " << hazardOrdinal << "\n";
        }
      });
      consumers |= userInfo.membership;
      opInfo.hazards |= userInfo.membership;
      opInfo.hazards |= userInfo.hazards;
    }
    llvm::BitVector candidates(builders.size(), /*t=*/true);
    candidates ^= opInfo.hazards;
    candidates |= consumers;
    candidates &= usableBuilders;

    // Prune candidates that do not have a compatible affinity.
    for (auto ordinal : candidates.set_bits()) {
      if (!IREE::Stream::AffinityAttr::areCompatible(
              affinityAttr, builders[ordinal]->affinity)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Candidate partition " << ordinal << " incompatible\n");
        candidates.reset(ordinal);
      }
    }

    // Prune candidates that are not on the same device
    if (device >= 0) {
      for (auto ordinal : candidates.set_bits()) {
        if (builders[ordinal]->device != device) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Candidate partition " << ordinal << " incompatible\n");
          candidates.reset(ordinal);
        }
      }
    }

    // If this op is not streamable then bail here; we've still setup the hazard
    // map for following iteration.
    auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(op);
    if (!streamableOp) {
      LLVM_DEBUG(llvm::dbgs() << "Not streamable (skip)\n");
      continue;
    }

    // First see which partitions are consuming this that we can also safely
    // move in to.
    consumers &= candidates;

    opInfo.membership.reserve(builders.size() + 1);
    opInfo.membership.resize(builders.size(), /*t=*/false);

    // If we have one or more consumers we should go into those first.
    if (consumers.any()) {
      // If we are a clonable op (like splat) clone us into every partition.
      // Otherwise we just pick the first we find (probably a bad heuristic).
      if (streamableOp.preferCloneToConsumers()) {
        for (auto consumerOrdinal : consumers.set_bits()) {
          LLVM_DEBUG(llvm::dbgs() << "Cloning into consumer partition "
                                  << consumerOrdinal << "\n");
          builders[consumerOrdinal]->ops.insert(&op);
          opInfo.membership.set(consumerOrdinal);
          opInfo.hazards.reset(consumerOrdinal);
        }
      } else {
        int consumerOrdinal = consumers.find_last();
        LLVM_DEBUG(llvm::dbgs() << "Moving into consumer partition "
                                << consumerOrdinal << "\n");
        builders[consumerOrdinal]->ops.insert(&op);
        opInfo.membership.set(consumerOrdinal);
        opInfo.hazards.reset(consumerOrdinal);
      }
      LLVM_DEBUG(llvm::dbgs() << "Handled streamable (continue)\n");
      continue;
    }

    // No consumers - if there's any candidate then we'll go into that.
    int firstCandidateOrdinal = candidates.find_first();
    if (firstCandidateOrdinal != -1) {
      LLVM_DEBUG(llvm::dbgs() << "Moving to first candidate partition "
                              << firstCandidateOrdinal << " (continue)\n");
      builders[firstCandidateOrdinal]->ops.insert(&op);
      opInfo.membership.set(firstCandidateOrdinal);
      opInfo.hazards.reset(firstCandidateOrdinal);
      continue;
    }

    // Mark the op as having hazards against all other partitions.
    if (!builders.empty()) {
      opInfo.hazards.set(0, builders.size() - 1);
    }

    // Create a new partition just for this op.
    opInfo.membership.resize(opInfo.membership.size() + 1, /*t=*/true);
    auto builder = std::make_unique<PartitionBuilder>();
    builder->ordinal = builders.size();
    builder->affinity = affinityAttr;
    builder->device = device;
    builder->ops.insert(&op);
    LLVM_DEBUG(llvm::dbgs()
               << "Created partition " << builder->ordinal << "\n");
    builders.push_back(std::move(builder));
    usableBuilders.resize(builders.size(), /*t=*/true);
  }

  // Emit partitions in forward order (as they are topologically sorted in
  // reverse order from our bottom-up walk).
  for (auto &builder : llvm::reverse(builders)) {
    Partition partition;

    SetVector<Value> consumedValues;
    SetVector<Value> producedValues;
    SetVector<Value> escapingValues;
    for (auto *op : llvm::reverse(builder->ops)) {
      for (auto operand : op->getOperands()) {
        consumedValues.insert(operand);
      }
      for (auto result : op->getResults()) {
        producedValues.insert(result);
        // TODO(benvanik): optimize this - creates n^2/nlogn behavior.
        for (auto user : result.getUsers()) {
          if (!builder->ops.contains(user)) {
            escapingValues.insert(result);
          }
        }
      }
    }
    consumedValues.set_subtract(producedValues);
    partition.ins = consumedValues;
    partition.outs = escapingValues;

    partition.ops = std::move(builder->ops);
    partitionSet.partitions.push_back(std::move(partition));
  }

  LLVM_DEBUG(partitionSet.dump(block->getParentOp()));

  return partitionSet;
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
