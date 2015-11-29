//===========================================================================
/*!
 * 
 *
 * \brief       Structure related type traits
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

#ifndef ABLAS_DETAIL_STRUCTURE_HPP
#define ABLAS_DETAIL_STRUCTURE_HPP

#include <cstddef>
#include "exception.hpp"

namespace aBLAS {


struct linear_structure{};
struct packed_structure{};
	
struct upper;
struct unit_upper;
	
///\brief Flag indicating that the matrix is lower triangular
struct lower{
	static const bool is_upper = false;
	static const bool is_unit = false;
	typedef upper transposed_orientation;
	
};
///\brief Flag indicating that the matrix is lower triangular and diagonal elements are to be assumed as 1
struct unit_lower{
	static const bool is_upper = false;
	static const bool is_unit = true;
	typedef unit_upper transposed_orientation;
};
	
///\brief Flag indicating that the matrix is upper triangular
struct upper{
	static const bool is_upper = true;
	static const bool is_unit = false;
	typedef lower transposed_orientation;
};
///\brief Flag indicating that the matrix is upper triangular and diagonal elements are to be assumed as 1
struct unit_upper{
	static const bool is_upper = true;
	static const bool is_unit = true;
	typedef unit_lower transposed_orientation;
};
	
struct column_major;

// This traits class defines storage layout and it's properties
// matrix (i,j) -> storage [i * size_i + j]
struct row_major:public linear_structure{
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef row_major orientation;
	typedef column_major transposed_orientation;
	template<class T>
	struct sparse_element{
		size_type i;
		size_type j;
		T value;
		
		bool operator<(sparse_element const& other)const{
			if(i == other.i)
				return j< other.j;
			else
				return i < other.i;
		}
		
	};

	// Indexing conversion to storage element
	static size_type element(size_type i, size_type size_i, size_type j, size_type size_j) {
		ABLAS_SIZE_CHECK(i < size_i);
		ABLAS_SIZE_CHECK(j < size_j);
		return i * size_j + j;
	}

	// Major and minor indices
	static size_type index_M(size_type index1, size_type /* index2 */) {
		return index1;
	}
	static size_type index_m(size_type /* index1 */, size_type index2) {
		return index2;
	}
	
	//from major and minor index to element index
	static size_type index_row(size_type major, size_type /*minor*/) {
		return major;
	}
	static size_type index_col(size_type /*major*/, size_type minor) {
		return minor;
	}
	
	static size_type stride1(size_type /*size_i*/, size_type size_j){
		return size_j;
	}
	static size_type stride2(size_type /*size_i*/, size_type /*size_j*/){
		return 1;
	}
	
	static size_type  triangular_index(size_type i, size_type j, size_type size,lower){
		return i*(i+1)/2+j; 
	}
	
	static size_type  triangular_index(size_type i, size_type j, size_type size,upper){
		return (i*(2*size-i+1))/2+j-i; 
	}
};

// This traits class defines storage layout and it's properties
// matrix (i,j) -> storage [i + j * size_i]
struct column_major:public linear_structure{
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef row_major transposed_orientation;
	template<class T>
	struct sparse_element{
		size_type i;
		size_type j;
		T value;
		
		bool operator<(sparse_element const& other)const{
			if(j == other.j)
				return i< other.i;
			else
				return j < other.j;
		}
		
	};

	// Indexing conversion to storage element
	static size_type element(size_type i, size_type size_i, size_type j, size_type size_j) {
		ABLAS_SIZE_CHECK(i < size_i);
		ABLAS_SIZE_CHECK(j < size_j);
		return i + j * size_i;
	}

	// Major and minor indices
	static size_type index_M(size_type /* index1 */, size_type index2) {
		return index2;
	}
	static size_type index_m(size_type index1, size_type /* index2 */) {
		return index1;
	}
	
	//from major and minor index to element index
	static size_type index_row(size_type /*major*/, size_type minor) {
		return minor;
	}
	static size_type index_col(size_type major, size_type /*minor*/) {
		return major;
	}
	
	static size_type stride1(size_type /*size_i*/, size_type /*size_j*/){
		return 1;
	}
	static size_type stride2(size_type size_i, size_type /*size_j*/){
		return size_i;
	}
	
	static size_type  triangular_index(size_type i, size_type j, size_type size,lower){
		return transposed_orientation::triangular_index(j,i,size,upper()); 
	}
	
	static size_type  triangular_index(size_type i, size_type j, size_type size,upper){
		return transposed_orientation::triangular_index(j,i,size,lower()); 
	}
};
struct unknown_orientation:public linear_structure
{typedef unknown_orientation transposed_orientation;};

//storage schemes for packed matrices
template<class Orientation, class TriangularType>
struct packed:public packed_structure{
	typedef  TriangularType triangular_type;
	typedef Orientation orientation;
	typedef packed<
		typename Orientation::transposed_orientation,
		typename TriangularType::transposed_orientation
	> transposed_orientation;
	
	typedef typename Orientation::size_type size_type;
	static bool non_zero(size_type i, size_type  j){
		return TriangularType::is_upper? j >= i: i >= j;
	}
	
	static size_type element(size_type i, size_type j, size_type size) {
		ABLAS_SIZE_CHECK(i <= size);
		ABLAS_SIZE_CHECK(j <= size);
		//~ ABLAS_SIZE_CHECK( non_zero(i,j));//lets end iterators fail!
		
		return orientation::triangular_index(i,j,size,TriangularType());
	}
	
	static size_type stride1(size_type size_i, size_type size_j){
		return orientation::stride1(size_i,size_j);
	}
	static size_type stride2(size_type size_i, size_type size_j){
		return orientation::stride2(size_i,size_j);
	}
};

}

#endif
