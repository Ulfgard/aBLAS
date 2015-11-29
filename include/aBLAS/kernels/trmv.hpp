/*!
 * 
 *
 * \brief       Dispatcher for the TRMV routine
 *
 * \author      O. Krause
 * \date        2015
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef ABLAS_KERNELS_TRMV_HPP
#define ABLAS_KERNELS_TRMV_HPP

#ifdef ABLAS_USE_CBLAS
#include "cblas/trmv.hpp"
#else
// if no bindings are included, we have to provide the default has_optimized_trmv
// otherwise the binding will take care of this
namespace aBLAS { namespace bindings{
template<class M, class V>
struct  has_optimized_trmv
: public boost::mpl::false_{};
}}
#endif

#include "default/trmv.hpp"

namespace aBLAS {namespace kernels{
	
///\brief Implements the TRiangular Matrix Vector Multiplication.
///
/// It computes b=alpha*Ab in place where A is a square lower or upper triangular matrix.
/// It can optionally assume that the diagonal is 1 and won't access the diagonal elements.
template <bool Upper,bool Unit,typename TriangularA, typename VecB>
void trmv(
	matrix_expression<TriangularA,cpu_tag> const &A, 
	vector_expression<VecB,cpu_tag>& b
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1() == b().size());
	
	bindings::trmv<Upper,Unit>(A,b,typename bindings::has_optimized_trmv<TriangularA, VecB>::type());
}

}}

#ifdef ABLAS_USE_OPEN_CL
#include "clBLAS/trmv.hpp"

namespace aBLAS {namespace kernels{
///\brief Implements the TRiangular Matrix Vector Multiplication.
///
/// If bindings are included and the matrix combination allow for a specific binding
/// to be applied, the binding is called automatically from {binding}/trmv.h
/// The kernels themselves are implemented in blas::bindings::trmv.
///
///Be aware that there are no default implentations for gpu operations.
///Not using any bindings will lead to a crash. This also holds for arguments
///with mixed types.
template <bool Upper,bool Unit,typename TriangularA, typename VecB>
void trmv(
	matrix_expression<TriangularA,gpu_tag> const &A, 
	vector_expression<VecB,gpu_tag>& b,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1() == b().size());
	
	static_assert(
		boost::is_same<typename TriangularA::value_type,typename  VecB::value_type>::value
		"Arguments must have the same value_type"
	);
	
	bindings::trmv<Upper,Unit>(A, b,queue);
}

}}

#endif

#endif
