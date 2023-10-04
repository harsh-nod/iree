#include "mlir/Dialect/NVGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/Transforms/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::iree_compiler {

/// The size of a shared memory line according to AMD documentation.
/// https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf
constexpr int64_t kSharedMemoryLineSizeBytes = 64;
/// We optimize for 64bit accesses, but this can be made an argument in the
/// future.
constexpr int64_t kDefaultVectorSizeBits = 64;

static Operation::operand_range getIndices(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getIndices();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getIndices();
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndices();
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndices();
  llvm_unreachable("unsupported op type");
}

static void setIndices(Operation *op, ArrayRef<Value> indices) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getIndicesMutable().assign(indices);
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getIndicesMutable().assign(indices);
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndicesMutable().assign(indices);
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndicesMutable().assign(indices);
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndicesMutable().assign(indices);
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndicesMutable().assign(indices);
  llvm_unreachable("unsupported op type");
}

static bool isSharedMemoryAddressSpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(memorySpace))
    return intAttr.getInt() == 3;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuAttr.getValue() == gpu::AddressSpace::Workgroup;
  return false;
}

static bool hasSharedMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  return isSharedMemoryAddressSpace(memorySpace);
}

/// Uses `srcIndexValue` to permute `tgtIndexValue` via
/// `result = xor(floordiv(srcIdxVal,permuteEveryN),
///               floordiv(tgtIdxVal,vectorSize)))
///            + tgtIdxVal % vectorSize`
/// This is done using an optimized sequence of `arith` operations.
static Value permuteVectorOffset(OpBuilder &b, Location loc,
                                 ArrayRef<Value> indices, MemRefType memrefTy,
                                 int64_t srcDim, int64_t tgtDim) {
  // Adjust the src index to change how often the permutation changes
  // if necessary.
  Value src = indices[srcDim];

  // We only want to permute every N iterations of the target dim where N is
  // ceil(sharedMemoryLineSizeBytes / dimSizeBytes(tgtDim)).
  const int64_t permuteEveryN = std::max<int64_t>(
      1, kSharedMemoryLineSizeBytes / ((memrefTy.getDimSize(tgtDim) *
                                        memrefTy.getElementTypeBitWidth()) /
                                       8));

  // clang-format off
  // Index bit representation (b0 = least significant bit) for dim(1)
  // of a `memref<?x?xDT>` is as follows:
  // N := log2(128/elementSizeBits)
  // M := log2(dimSize(1))
  // then
  // bits[0:N] = sub-vector element offset
  // bits[N:M] = vector index
  // clang-format on
  int64_t n =
      llvm::Log2_64(kDefaultVectorSizeBits / memrefTy.getElementTypeBitWidth());
  int64_t m = llvm::Log2_64(memrefTy.getDimSize(tgtDim));

  // Capture bits[0:(M-N)] of src by first creating a (M-N) mask.
  int64_t mask = (1LL << (m - n)) - 1;
  if (permuteEveryN > 1)
    mask = mask << llvm::Log2_64(permuteEveryN);
  Value srcBits = b.create<arith::ConstantIndexOp>(loc, mask);
  srcBits = b.create<arith::AndIOp>(loc, src, srcBits);

  // Use the src bits to permute the target bits b[N:M] containing the
  // vector offset.
  if (permuteEveryN > 1) {
    int64_t shlBits = n - llvm::Log2_64(permuteEveryN);
    if (shlBits > 0) {
      Value finalShiftVal = b.create<arith::ConstantIndexOp>(loc, shlBits);
      srcBits = b.createOrFold<arith::ShLIOp>(loc, srcBits, finalShiftVal);
    } else if (shlBits < 0) {
      Value finalShiftVal = b.create<arith::ConstantIndexOp>(loc, -1 * shlBits);
      srcBits = b.createOrFold<arith::ShRUIOp>(loc, srcBits, finalShiftVal);
    }
  } else {
    Value finalShiftVal = b.create<arith::ConstantIndexOp>(loc, n);
    srcBits = b.createOrFold<arith::ShLIOp>(loc, srcBits, finalShiftVal);
  }

  Value permutedVectorIdx =
      b.create<arith::XOrIOp>(loc, indices[tgtDim], srcBits);
  return permutedVectorIdx;
}

static void transformIndices(OpBuilder &builder, Location loc,
                             SmallVector<Value, 4> &indices,
                             MemRefType memrefTy, int64_t srcDim,
                             int64_t tgtDim) {
  indices[tgtDim] =
      permuteVectorOffset(builder, loc, indices, memrefTy, srcDim, tgtDim);
}

/// Return all operations within `parentOp` that read from or write to
/// `shmMemRef`.
static LogicalResult
getShmReadAndWriteOps(Operation *parentOp, Value shmMemRef,
                      SmallVector<Operation *, 16> &readOps,
                      SmallVector<Operation *, 16> &writeOps) {
  parentOp->walk([&](Operation *op) {
    MemoryEffectOpInterface iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface)
      return;
    std::optional<MemoryEffects::EffectInstance> effect =
        iface.getEffectOnValue<MemoryEffects::Read>(shmMemRef);
    if (effect) {
      readOps.push_back(op);
      return;
    }
    effect = iface.getEffectOnValue<MemoryEffects::Write>(shmMemRef);
    if (effect)
      writeOps.push_back(op);
  });

  // Restrict to a supported set of ops. We also require at least 2D access,
  // although this could be relaxed.
  if (llvm::any_of(readOps, [](Operation *op) {
        return !isa<memref::LoadOp, vector::LoadOp, vector::TransferReadOp>(op) ||
               getIndices(op).size() < 2;
      }))
    return failure();
  if (llvm::any_of(writeOps, [](Operation *op) {
        return !isa<memref::StoreOp, vector::StoreOp, vector::TransferWriteOp>(
                   op) ||
               getIndices(op).size() < 2;
      }))
    return failure();

  return success();
}

mlir::LogicalResult
optimizeSharedMemoryReadsAndWrites(Operation *parentOp, Value memrefValue) {
  auto memRefType = dyn_cast<MemRefType>(memrefValue.getType());
  if (!memRefType || !hasSharedMemoryAddressSpace(memRefType))
    return failure();

  // Abort if the given value has any sub-views; we do not do any alias
  // analysis.
  bool hasSubView = false;
  parentOp->walk([&](memref::SubViewOp subView) { hasSubView = true; });
  if (hasSubView)
    return failure();

  // Check if this is necessary given the assumption of 128b accesses:
  // If dim[rank-1] is small enough to fit 8 rows in a 128B line.
  const int64_t rowSize = memRefType.getDimSize(memRefType.getRank() - 1);
  const int64_t rowsPerLine =
      (8 * kSharedMemoryLineSizeBytes / memRefType.getElementTypeBitWidth()) /
      rowSize;
  const int64_t threadGroupSize =
      1LL << (7 - llvm::Log2_64(kDefaultVectorSizeBits / 8));
  if (rowsPerLine >= threadGroupSize)
    return failure();

  // Get sets of operations within the function that read/write to shared
  // memory.
  SmallVector<Operation *, 16> shmReadOps;
  SmallVector<Operation *, 16> shmWriteOps;
  if (failed(getShmReadAndWriteOps(parentOp, memrefValue, shmReadOps,
                                   shmWriteOps)))
    return failure();

  if (shmReadOps.empty() || shmWriteOps.empty())
    return failure();

  OpBuilder builder(parentOp->getContext());

  int64_t tgtDim = memRefType.getRank() - 1;
  int64_t srcDim = memRefType.getRank() - 2;

  // Transform indices for the ops writing to shared memory.
  while (!shmWriteOps.empty()) {
    Operation *shmWriteOp = shmWriteOps.back();
    shmWriteOps.pop_back();
    builder.setInsertionPoint(shmWriteOp);

    auto indices = getIndices(shmWriteOp);
    SmallVector<Value, 4> transformedIndices(indices.begin(), indices.end());
    transformIndices(builder, shmWriteOp->getLoc(), transformedIndices,
                     memRefType, srcDim, tgtDim);
    setIndices(shmWriteOp, transformedIndices);
  }

  // Transform indices for the ops reading from shared memory.
  while (!shmReadOps.empty()) {
    Operation *shmReadOp = shmReadOps.back();
    shmReadOps.pop_back();
    builder.setInsertionPoint(shmReadOp);

    auto indices = getIndices(shmReadOp);
    SmallVector<Value, 4> transformedIndices(indices.begin(), indices.end());
    transformIndices(builder, shmReadOp->getLoc(), transformedIndices,
                     memRefType, srcDim, tgtDim);
    setIndices(shmReadOp, transformedIndices);
  }

  return success();
}

void optimizeSharedMemoryReadsAndWrites(func::FuncOp funcOp) {
  SmallVector<memref::AllocOp> shmAllocOps;
  funcOp.walk([&](memref::AllocOp allocOp) {
    if (!hasSharedMemoryAddressSpace(allocOp.getType()))
      return;
    shmAllocOps.push_back(allocOp);
  });
  for (auto allocOp : shmAllocOps) {
    if (failed(optimizeSharedMemoryReadsAndWrites(funcOp,
                                                  allocOp.getMemref())))
      return;
  }
}

} // namespace