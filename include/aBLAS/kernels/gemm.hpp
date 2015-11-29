/*!
 * 
 *
 * \brief       Dispatcher for the GEMM routine
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
#ifndef ABLAS_KERNELS_GEMM_HPP
#define ABLAS_KERNELS_GEMM_HPP

#ifdef ABLAS_USE_CBLAS
#include "cblas/gemm.hpp"
#else

//if no bindings are included, we have to provide the default has_optimized_gemm otherwise the binding will take care of this
namespace aBLAS { namespace bindings{
template<class M1, class M2, class M3>
struct  has_optimized_gemm
: public boost::mpl::false_{};
}}
#endif

#include "default/gemm.hpp"

namespace aBLAS {namespace kernels{
	
///\brief Well known GEneral Matrix-Matrix product kernel M+=alpha*E1*E2.
///
/// If bindings are included and the matrix combination allow for a specific binding
/// to be applied, the binding is called automatically from {binding}/gemm.h
/// otherwise default/gemm.h is used which is fully implemented for all dense/sparse combinations.
/// if a combination is optimized, bindings::has_optimized_gemm<M,E1,E2>::type evaluates to boost::mpl::true_
/// The kernels themselves are implemented in blas::bindings::gemm.
template<class M, class E1, class E2>
void gemm(
	matrix_expression<E1,cpu_tag> const& e1,
	matrix_expression<E2,cpu_tag> const& e2,
	matrix_expression<M,cpu_tag>& m,
	typename M::value_type alpha
) {
	ABLAS_SIZE_CHECK(m().size1() == e1().size1());
	ABLAS_SIZE_CHECK(m().size2() == e2().size2());
	ABLAS_SIZE_CHECK(e1().size2() == e2().size1());
	
	bindings::gemm(
		e1, e2, m,alpha,
		typename bindings::has_optimized_gemm<M,E1,E2>::type()
	);
}

}}

#ifdef SHARK_USE_OPEN_CL
#include "clBLAS/gemm.hpp"

namespace aBLAS{namespace kernels{
///\brief Well known GEneral Matrix-Matrix product kernel M+=alpha*E1*E2 for gpu expressions.
///
/// If bindings are included and the matrix combination allow for a specific binding
/// to be applied, the binding is called automatically from {binding}/gemm.h
/// The kernels themselves are implemented in blas::bindings::gemm.
///
///Be aware that there are no default implentations for gpu operations.
///Not using any bindings will lead to a crash. This also holds for arguments
///with mixed types.
template<class M, class E1, class E2>
void gemm(
	matrix_expression<E1,gpu_tag> const& e1,
	matrix_expression<E2,gpu_tag> const& e2,
	matrix_expression<M,gpu_tag>& m,
	typename M::value_type alpha,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
) {
	ABLAS_SIZE_CHECK(m().size1() == e1().size1());
	ABLAS_SIZE_CHECK(m().size2() == e2().size2());
	ABLAS_SIZE_CHECK(e1().size2() == e2().size1());
	
	static_assert(
		boost::is_same<M::value_type,E1::value_type>::value
		&& boost::is_same<MatrA::value_type,E2::value_type>::value,
		"Arguments must have the same value_type"
	);
	
	bindings::gemm(e1, e2, m,alpha,queue);
}

}}

#endif
#endif