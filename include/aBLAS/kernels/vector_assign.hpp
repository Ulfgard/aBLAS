/*!
 * 
 *
 * \brief       Dispatcher and Implementation for vector assignment operations
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
#ifndef ABLAS_KERNELS_VECTOR_ASSIGN_HPP
#define ABLAS_KERNELS_VECTOR_ASSIGN_HPP

#include "../detail/functional.hpp"
#include "../expression_types.hpp"

namespace aBLAS{
namespace kernels{

////////////////////////////////////////////
//assignment of constant value with functor
////////////////////////////////////////////
template<template <class T1, class T2> class F, class V>
void assign(vector_expression<V,cpu_tag>& v, typename V::value_type t) {
	typedef F<typename V::iterator::reference, typename V::value_type> Function;
	Function f(typename V::value_type(1));
	typedef typename V::iterator iterator;
	iterator end = v().end();
	for (iterator it = v().begin(); it != end; ++it){
		f(*it, t);
	}
}

////////////////////////////////////////////
//assignment with functor
////////////////////////////////////////////

//dense dense case
template<template <class T1, class T2> class F, class V, class E>
void assign(
	vector_expression<V,cpu_tag>& v,
	vector_expression<E,cpu_tag> const& e,
	typename V::value_type alpha,
	dense_random_access_iterator_tag, dense_random_access_iterator_tag
) {
	F<typename V::reference, typename E::value_type> f(alpha);
	typename V::iterator end_v = v().end();
	typename V::iterator pos_v =v().begin();
	typename E::const_iterator pos_e =e().begin();
	for(; pos_v != end_v; ++pos_v,++pos_e){
		f(*pos_v,*pos_e);
	}
}

// Dispatcher
template<template <class T1, class T2> class F, class V, class E>
void assign(
	vector_expression<V,cpu_tag>& v,
	vector_expression<E,cpu_tag> const&e,
	typename V::value_type alpha
) {
	ABLAS_SIZE_CHECK(v().size() == e().size());
	typedef typename V::const_iterator::iterator_category CategoryV;
	typedef typename E::const_iterator::iterator_category CategoryE;
	assign<F>(v(), e(), alpha, CategoryV(),CategoryE());
}

}}
#endif