/*!
 * 
 *
 * \brief       Default implementation for the GEMV routine
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
#ifndef ABLAS_KERNELS_DEFAULT_GEMV_HPP
#define ABLAS_KERNELS_DEFAULT_GEMV_HPP

#include "../../matrix_proxy.hpp"
#include "../dot.hpp"
#include <boost/mpl/bool.hpp>

namespace aBLAS {namespace bindings {
	
//row major can be further reduced to inner_prod()
template<class ResultV, class M, class V>
void gemv_impl(
	matrix_expression<M,cpu_tag> const& A,
	vector_expression<V,cpu_tag> const& x,
	vector_expression<ResultV,cpu_tag>& result, 
	typename ResultV::value_type alpha,
	row_major
) {
	typedef typename ResultV::value_type value_type;
	for(std::size_t i = 0; i != A().size1();++i){
		value_type value = 0;
		kernels::dot(row(A,i),x,value);
		result()(i) += alpha* value;
	}
}

//column major is implemented by computing a linear combination of matrix-rows 
template<class ResultV, class M, class V>
void gemv_impl(
	matrix_expression<M,cpu_tag> const& A,
	vector_expression<V,cpu_tag> const& x,
	vector_expression<ResultV,cpu_tag>& result,
	typename ResultV::value_type alpha,
	column_major
) {
	typedef typename V::const_iterator iterator;
	typedef typename ResultV::value_type value_type;
	iterator end = x().end();
	for(iterator it = x().begin(); it != end; ++it) {
		value_type multiplier = alpha * (*it);
		//fixme: for sparse result vectors, this might hurt.
		kernels::assign<scalar_plus_assign>(result, column(A(), it.index()),multiplier);
	}
}

//unknown orientation is dispatched to row_major
template<class ResultV, class M, class V>
void gemv_impl(
	matrix_expression<M,cpu_tag> const& A,
	vector_expression<V,cpu_tag> const& x,
	vector_expression<ResultV,cpu_tag>& result,
	typename ResultV::value_type alpha,
	unknown_orientation
) {
	gemv_impl(A,x,result,alpha,row_major());
}

// result += alpha * A * x
template<class ResultV, class M, class V>
void gemv(
	matrix_expression<M,cpu_tag> const& A,
        vector_expression<V,cpu_tag> const& x,
        vector_expression<ResultV,cpu_tag>& result, 
	typename ResultV::value_type alpha,
	boost::mpl::false_
) {
	typedef typename M::orientation orientation;

	gemv_impl(A, x, result, alpha, orientation());
}

}}
#endif
