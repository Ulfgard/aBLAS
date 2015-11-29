//===========================================================================
/*!
 * 
 *
 * \brief       Expression Tags for dispatching
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

#ifndef ABLAS_DETAIL_TAGS_HPP
#define ABLAS_DETAIL_TAGS_HPP

#include <iterator>

namespace aBLAS {

// Storage tags -- hierarchical definition of storage characteristics
// this gives the real storage layout of the matix in memory
// packed_tag ->BLAS packed format and supports packed interface
// dense_tag -> dense storage scheme an dense interface supported
// sparse_tag -> sparse storage scheme and supports sparse interface.
// unknown_storage_tag -> no known storage scheme, only supports basic interface
struct unknown_storage_tag {};
struct sparse_tag:public unknown_storage_tag{};
struct dense_tag: public unknown_storage_tag{};
struct packed_tag: public unknown_storage_tag{};

//device tags
//cpu_tag -> expression resides on cpu
//gpu_tag -> expression resides on gpu
struct cpu_tag{};
struct gpu_tag{};
	
//evaluation tags
// elementwise_tag -> the expression can directly be evaluated using the iterators and elementwise access
// blockwise_tag -> the expression can only be evaluted using the assign_to/plus_assign_to interface
struct elementwise_tag{};
struct blockwise_tag{};

namespace detail{
	template<class S1, class S2>
	struct evaluation_restrict_traits {
		typedef S1 type;
	};
	template<>
	struct evaluation_restrict_traits<elementwise_tag, blockwise_tag > {
		typedef blockwise_tag type;
	};
}

template<class E1, class E2>
struct evaluation_restrict_traits: public detail::evaluation_restrict_traits<
	typename E1::evaluation_category,
	typename E2::evaluation_category
>{};
	
// Iterator tags -- hierarchical definition of storage characteristics
struct sparse_bidirectional_iterator_tag: public std::bidirectional_iterator_tag{};
struct packed_random_access_iterator_tag: public std::random_access_iterator_tag{};
struct dense_random_access_iterator_tag: public packed_random_access_iterator_tag{};

}

#endif
