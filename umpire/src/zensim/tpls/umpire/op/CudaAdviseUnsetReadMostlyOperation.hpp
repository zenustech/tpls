//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaAdviseUnsetReadMostlyOperation_HPP
#define UMPIRE_CudaAdviseUnsetReadMostlyOperation_HPP

#include "zensim/tpls/umpire/op/MemoryOperation.hpp"

namespace umpire {
namespace op {

class CudaAdviseUnsetReadMostlyOperation : public MemoryOperation {
 public:
  /*!
   * @copybrief MemoryOperation::apply
   *
   * Uses cudaMemAdvise to set data as "read mostly" on the appropriate device.
   *
   * @copydetails MemoryOperation::apply
   */
  void apply(void* src_ptr, util::AllocationRecord* src_allocation, int val,
             std::size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaAdviseUnsetReadMostlyOperation_HPP
