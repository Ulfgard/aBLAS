//===========================================================================
/*!
 * 
 *
 * \brief       Basic implementation for assigning two CPU matrices
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
//===========================================================================
#ifndef ABLAS_KERNELS_MATRIX_ASSIGN_HPP
#define ABLAS_KERNELS_MATRIX_ASSIGN_HPP

#include "../detail/traits.hpp"
#include <algorithm>
namespace aBLAS{
	
//////////////////////////////////////////////////////
////Scalar Assignment to Matrix
/////////////////////////////////////////////////////

namespace kernels{

template<template <class T1, class T2> class F, class M, class Orientation >
void assign(
	matrix_expression<M,cpu_tag> &m, 
	typename M::value_type t, 
	Orientation,
	dense_random_access_iterator_tag
){
	std::size_t majorSize = Orientation::index_M(m().size1(),m().size2());
	F<typename M::reference, typename M::value_type> f(typename M::value_type(1));
	for(std::size_t i = 0; i != majorSize; ++i){
		std::for_each(major_begin(m,i),major_end(m,i),[t,&f](typename M::reference& val){f(val,t);});
	}
}

// Dispatcher
template<template <class T1, class T2> class F, class M>
void assign(
	matrix_expression<M,cpu_tag> &m, 
	typename M::value_type t
){
	typedef typename M::orientation orientation;
	typedef typename major_iterator<M>::type::iterator_category category;
	assign<F>(m, t, orientation(),category());
}

///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing =, +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////

template<template <class, class> class F, class M, class E, class Orientation>
void assign(
	matrix_expression<M,cpu_tag> &m, 
	matrix_expression<E,cpu_tag> const& e,
	typename M::value_type alpha,
	Orientation, Orientation,
	dense_random_access_iterator_tag,
	dense_random_access_iterator_tag
) {
	std::size_t size_M = Orientation::index_M(m().size1(),m().size2());
	F<typename M::reference, typename E::value_type> f(alpha);
	typedef typename major_iterator<M>::type M_iterator;
	typedef typename major_iterator<E const>::type E_iterator;
	for(std::size_t i = 0; i != size_M; ++i){
		M_iterator end_M = major_end(m,i);
		M_iterator pos_M =major_begin(m,i);
		E_iterator pos_E =major_begin(e,i);
		for(; pos_M != end_M; ++pos_M,++pos_E){
			f(*pos_M,*pos_E);
		}
	}
}
template<template <class, class> class F, class M, class E, class Orientation>
void assign(
	matrix_expression<M,cpu_tag> &m, 
	matrix_expression<E,cpu_tag> const& e,
	typename M::value_type alpha,
	Orientation, typename Orientation::transposed_orientation,
	dense_random_access_iterator_tag,
	dense_random_access_iterator_tag
) {
	typedef typename M::size_type size_type;
	//compute blockwise the assignment blockwise using an intermediate blockStorage
	//this is chosen as the blockStorage can be kept in L1 cache and thus we do not have
	//to worry about access speed of it and thus can first write the block in a way that is
	//quick for e and then write the block to m that is quick for m.
	size_type const blockSize = 16;
	typename M::value_type blockStorage[blockSize][blockSize];
	
	size_type size_M = Orientation::index_M(m().size1(),m().size2());
	size_type size_m = Orientation::index_m(m().size1(),m().size2());
	F<typename M::reference, typename E::value_type> f(alpha);
	for (size_type iblock = 0; iblock < size_M; iblock += blockSize){
		for (size_type jblock = 0; jblock < size_m; jblock += blockSize){
			std::size_t blockSizei = std::min(blockSize,size_M-iblock);
			std::size_t blockSizej = std::min(blockSize,size_m-jblock);
			
			//read block values into the block by iterating over the fast direction of e
			//as we iteratore over the major and minor directions, we have to convert the indexes to row and column
			for (size_type j = 0; j < blockSizej; ++j){
				for (size_type i = 0; i < blockSizei; ++i){
					size_type block_row = Orientation::index_row(i,j);
					size_type block_col = Orientation::index_col(i,j);
					size_type e_row = Orientation::index_row(iblock+i,jblock+j);
					size_type e_col = Orientation::index_col(iblock+i,jblock+j);
					blockStorage[block_row][block_col] = e()(e_row,e_col);
				}
			}
			
			//copy block in into m
			for (size_type i = 0; i < blockSizei; ++i){
				for (size_type j = 0; j < blockSizej; ++j){
					size_type block_row = Orientation::index_row(i,j);
					size_type block_col = Orientation::index_col(i,j);
					size_type m_row = Orientation::index_row(iblock+i,jblock+j);
					size_type m_col = Orientation::index_col(iblock+i,jblock+j);
					f(m()(m_row,m_col), blockStorage[block_row][block_col]);
				}
			}
		}
	}
}

//general dispatcher: if the second argument has an unknown orientation
// it is chosen the same as the first one
template<template <class, class> class F, class M, class E, class TagE, class TagM>
void assign(
	matrix_expression<M,cpu_tag> &m, 
	matrix_expression<E,cpu_tag> const& e,
	typename M::value_type alpha,
	row_major, unknown_orientation ,TagE tagE, TagM tagM
) {
	assign<F> (m,e,alpha, row_major(),row_major(),tagE,tagM);
}

//Dispatcher
template<template <class,class> class F, class M, class E>
void assign(matrix_expression<M,cpu_tag> &m, const matrix_expression<E,cpu_tag> &e, typename M::value_type alpha) {
	ABLAS_SIZE_CHECK(m().size1()  == e().size1());
	ABLAS_SIZE_CHECK(m().size2()  == e().size2());
	typedef typename M::orientation MOrientation;
	typedef typename E::orientation EOrientation;
	typedef typename major_iterator<M>::type::iterator_category MCategory;
	typedef typename major_iterator<E>::type::iterator_category ECategory;
	
	assign<F>(m, e, alpha, MOrientation(),EOrientation(), MCategory(), ECategory());
}

}}

#endif
