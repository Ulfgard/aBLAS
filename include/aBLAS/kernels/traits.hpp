/*!
 * 
 *
 * \brief       Storage Traits used in the bindings
 *
 * \author      O. Krause
 * \date        2013
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

#ifndef ABLAS_KERNELS_TRAITS_HPP
#define ABLAS_KERNELS_TRAITS_HPP

#include "../detail/traits.hpp"
#include <boost/type_traits/remove_reference.hpp>

namespace aBLAS {namespace bindings{ namespace traits {
	
///////////////Vector Traits//////////////////////////
	
template <typename V,class Device>
typename V::difference_type stride(vector_expression<V,Device> const&v) { 
	return v().stride();
}

template <typename V>
typename V::storage_type storage(vector_expression<V,cpu_tag>& v) { 
	return v().storage().data()+v().offset();
}
template <typename V>
typename V::const_storage_type storage(vector_expression<V,cpu_tag> const& v) { 
	return v().storage()+v().offset();
}

//////////////////Matrix Traits/////////////////////
template <typename M,class Device>
typename M::difference_type stride1(matrix_expression<M,Device> const& m) { 
	return m().stride1();
}
template <typename M,class Device>
typename M::difference_type stride2(matrix_expression<M,Device> const& m) { 
	return m().stride2();
}

template <typename M>
typename M::storage_type storage(matrix_expression<M,cpu_tag>& m) { 
	return m().storage().data()+m().offset();
}
template <typename M>
typename M::const_storage_type storage(matrix_expression<M,cpu_tag> const& m) { 
	return m().storage().data()+m().offset();
}

template <typename M,class Device>
typename M::difference_type leading_dimension(matrix_expression<M,Device> const& m) {
	return  M::orientation::index_M(stride1(m),stride2(m));
}

template<class M1, class M2,class Device1, class Device2>
bool same_orientation(matrix_expression<M1,Device1> const& m1, matrix_expression<M2,Device2> const& m2){
	return boost::is_same<typename M1::orientation,typename M2::orientation>::value;
}


}}}
#endif