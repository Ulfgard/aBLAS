/*!
 * 
 *
 * \brief       Dispatcher for the DOT routine
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
#ifndef ABLAS_KERNELS_DOT_HPP
#define ABLAS_KERNELS_DOT_HPP

#include "default/dot.hpp"


// if no bindings are included, we have to provide the default has_optimized_dot
// otherwise the binding will take care of this
namespace aBLAS { namespace bindings{
template<class V1, class V2,class result_type>
struct  has_optimized_dot
: public boost::mpl::false_{};
}}

namespace aBLAS {namespace kernels{
	
///\brief Well known dot-product r=<e1,e2>=sum_i e1_i*e2_i.
template<class E1, class E2,class result_type>
void dot(
	vector_expression<E1,cpu_tag> const& e1,
	vector_expression<E2,cpu_tag> const& e2,
	result_type& result
) {
	ABLAS_SIZE_CHECK(e1().size() == e2().size());
	
	bindings::dot(
		e1, e2,result,
		typename bindings::has_optimized_dot<E1,E2,result_type>::type()
	);
}

}}
#endif