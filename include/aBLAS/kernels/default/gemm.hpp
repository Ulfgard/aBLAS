/*!
 * 
 *
 * \brief       Default implementation for the GEMM routine
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

#ifndef ABLAS_KERNELS_DEFAULT_GEMM_HPP
#define ABLAS_KERNELS_DEFAULT_GEMM_HPP

#include "../gemv.hpp"
#include "../../matrix_proxy.hpp"
#include "../../vector.hpp"
#include <boost/mpl/bool.hpp>

namespace aBLAS { namespace bindings {
	
	
//general case: result and first argument row_major (2.)
//=> compute as a sequence of matrix-vector products over the rows of the first argument
template<class M, class E1, class E2, class Orientation2,class Tag1,class Tag2>
void gemm_impl(
	matrix_expression<E1,cpu_tag> const& e1,
	matrix_expression<E2,cpu_tag> const& e2,
	matrix_expression<M,cpu_tag>& m,
	typename M::value_type alpha,
	row_major, row_major, Orientation2, 
	Tag1, Tag2
) {
	for (std::size_t i = 0; i != e1().size1(); ++i) {
		matrix_row<M> mat_row(m(),i);
		kernels::gemv(trans(e2),row(e1,i),mat_row,alpha);
	}
}

//case: result and second argument row_major, first argument dense column major (3.2)
//=> compute as a sequence of outer products. 
template<class M, class E1, class E2,class Tag>
void gemm_impl(
	matrix_expression<E1,cpu_tag> const& e1,
	matrix_expression<E2,cpu_tag> const& e2,
	matrix_expression<M,cpu_tag>& m,
	typename M::value_type alpha,
	row_major, column_major, row_major,
	packed_random_access_iterator_tag, Tag
) {
	for (std::size_t k = 0; k != e1().size2(); ++k) {
		matrix_row<E2> e2_row(e2(),k);
		
		typename E1::const_column_iterator e1_col_pos = e1().column_begin(k);
		typename E1::const_column_iterator e1_col_end = e1().column_end(k);
		for(;e1_col_pos != e1_col_end; ++e1_col_pos){
			matrix_row<M> m_row(m(),e1_col_pos.index());
			kernels::assign<scalar_plus_assign>(m_row,e2_row,alpha* *e1_col_pos);
		}
	}
}


template<class M, class E1, class E2>
void gemm_impl(
	matrix_expression<E1,cpu_tag> const& e1,
	matrix_expression<E2,cpu_tag> const& e2,
	matrix_expression<M,cpu_tag>& m,
	typename M::value_type alpha,
	row_major r, column_major, column_major, 
	dense_random_access_iterator_tag t, dense_random_access_iterator_tag
) {
	//compute blockwise and write the transposed block.
	std::size_t blockSize = 24;
	typedef typename M::value_type value_type;
	typedef typename matrix_temporary<M>::type BlockStorage;
	BlockStorage blockStorage(blockSize,blockSize);
	
	typedef typename M::size_type size_type;
	size_type size1 = m().size1();
	size_type size2 = m().size2();
	for (size_type i = 0; i < size1; i+= blockSize){
		for (size_type j = 0; j < size2; j+= blockSize){
			std::size_t blockSizei = std::min(blockSize,size1-i);
			std::size_t blockSizej = std::min(blockSize,size2-j);
			matrix_range<matrix<value_type> > transBlock=subrange(blockStorage,0,blockSizej,0,blockSizei);
			kernels::assign<scalar_assign>(transBlock,value_type /* zero */());
			//reduce to all row-major case by using
			//A_ij=B^iC_j <=> A_ij^T = (C_j)^T (B^i)^T  
			gemm_impl(
				trans(columns(e2,j,j+blockSizej)),
				trans(rows(e1,i,i+blockSizei)),
				transBlock,alpha,
				r,r,r,//all row-major
				t,t //both targets are dense
			);
			//write transposed block to the matrix
			matrix_range<M> m_block = subrange(m,i,i+blockSizei,j,j+blockSizej);
			kernels::assign<scalar_plus_assign>(m_block,trans(transBlock),value_type(1));
		}
	}
}

//general case: column major result case (1.0)
//=> transformed to row_major using A=B*C <=> A^T = C^T B^T
template<class M, class E1, class E2, class Orientation1, class Orientation2, class Tag1, class Tag2>
void gemm_impl(
	matrix_expression<E1,cpu_tag> const& e1,
	matrix_expression<E2,cpu_tag> const& e2,
	matrix_expression<M,cpu_tag>& m,
	typename M::value_type alpha,
	column_major, Orientation1, Orientation2, 
	Tag1, Tag2
){
	matrix_transpose<M> transposedM(m());
	typedef typename Orientation1::transposed_orientation transpO1;
	typedef typename Orientation2::transposed_orientation transpO2;
	gemm_impl(trans(e2),trans(e1),transposedM,alpha,row_major(),transpO2(),transpO1(), Tag2(),Tag1());
}

//dispatcher
template<class M, class E1, class E2>
void gemm(
	matrix_expression<E1,cpu_tag> const& e1,
	matrix_expression<E2,cpu_tag> const& e2,
	matrix_expression<M,cpu_tag>& m,
	typename M::value_type alpha,
	boost::mpl::false_
) {
	typedef typename M::orientation ResultOrientation;
	typedef typename E1::orientation E1Orientation;
	typedef typename E2::orientation E2Orientation;
	typedef typename major_iterator<E1>::type::iterator_category E1Category;
	typedef typename major_iterator<E2>::type::iterator_category E2Category;
	
	gemm_impl(e1, e2, m,alpha,
		ResultOrientation(),E1Orientation(),E2Orientation(),
		E1Category(),E2Category()
	);
}

}}

#endif
