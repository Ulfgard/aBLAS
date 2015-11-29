//===========================================================================
/*!
 * 
 *
 * \brief       Traits classes and other type magic
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

#ifndef ABLAS_DETAIL_TRAITS_HPP
#define ABLAS_DETAIL_TRAITS_HPP

#include <complex>

#include "tags.hpp"
#include "iterator.hpp"
#include "structure.hpp"
#include "returntype_deduction.hpp"
#include "../expression_types.hpp"

#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/and.hpp>

#include <boost/range/iterator.hpp>

namespace aBLAS{
template<class M>
struct row_iterator: public boost::mpl::if_<
	boost::is_const<M>,
	typename M::const_row_iterator,
	typename M::row_iterator
>{};
	
template<class M>
struct column_iterator: public boost::mpl::if_<
	boost::is_const<M>,
	typename M::const_column_iterator,
	typename M::column_iterator
>{};

template<class Matrix> 
struct major_iterator:public boost::mpl::if_<
	boost::is_same<typename Matrix::orientation, column_major>,
	typename column_iterator<Matrix>::type,
	typename row_iterator<Matrix>::type
>{};
	
namespace detail{
	template<class M>
	typename column_iterator<M>::type major_begin(M& m,std::size_t i, column_major){
		return m.column_begin(i);
	}
	template<class M>
	typename row_iterator<M>::type major_begin(M& m,std::size_t i, row_major){
		return m.row_begin(i);
	}
	template<class M>
	typename column_iterator<M>::type major_end(M& m,std::size_t i, column_major){
		return m.column_end(i);
	}
	template<class M>
	typename row_iterator<M>::type major_end(M& m,std::size_t i, row_major){
		return m.row_end(i);
	}
}

template<class M>
typename major_iterator<M const>::type major_begin(matrix_expression<M, cpu_tag> const& m, std::size_t i){
	return detail::major_begin(m(),i, typename M::orientation());
}
template<class M>
typename major_iterator<M const>::type major_end(matrix_expression<M, cpu_tag> const& m, std::size_t i){
	return detail::major_end(m(),i, typename M::orientation());
}
template<class M>
typename major_iterator<M>::type major_begin(matrix_expression<M, cpu_tag>& m, std::size_t i){
	return detail::major_begin(m(),i, typename M::orientation());
}
template<class M>
typename major_iterator<M>::type major_end(matrix_expression<M, cpu_tag>& m, std::size_t i){
	return detail::major_end(m(),i, typename M::orientation());
}
	
template<class T>
struct real_traits{
	typedef T type;
};

template<class T>
struct real_traits<std::complex<T> >{
	typedef T type;
};

// Use Joel de Guzman's return type deduction
// uBLAS assumes a common return type for all binary arithmetic operators
template<class X, class Y>
struct promote_traits {
	typedef type_deduction_detail::base_result_of<X, Y> base_type;
	static typename base_type::x_type x;
	static typename base_type::y_type y;
	static const std::size_t size = sizeof(
	        type_deduction_detail::test<
	        typename base_type::x_type
	        , typename base_type::y_type
	        >(x + y)     // Use x+y to stand of all the arithmetic actions
	        );

	static const std::size_t index = (size / sizeof(char)) - 1;
	typedef typename boost::mpl::at_c<
	typename base_type::types, index>::type id;
	typedef typename id::type promote_type;
};
// special case for bools. b1+b2 creates a boolean return type - which does not make sense
// for example when summing bools! therefore we use a signed int type
template<>
struct promote_traits<bool, bool> {
	typedef int promote_type;
};


template<class E>
struct closure: public boost::mpl::if_<
	boost::is_const<E>,
	typename E::const_closure_type,
	typename E::closure_type
>{};

template<class E>
struct reference: public boost::mpl::if_<
	boost::is_const<E>,
	typename E::const_reference,
	typename E::reference
>{};

template<class E>
struct storage: public boost::mpl::if_<
	boost::is_const<E>,
	typename E::const_storage_type,
	typename E::storage_type
>{};
//~ template<class E>
//~ struct index_pointer: public boost::mpl::if_<
	//~ boost::is_const<E>,
	//~ typename E::const_index_pointer,
	//~ typename E::index_pointer
//~ >{};

///\brief Determines a good vector type storing an expression returning values of type T and having a certain iterator category.
template<class ValueType, class IteratorTag, class Device>
struct vector_temporary_type;
///\brief Determines a good vector type storing an expression returning values of type T and having a certain iterator category.
template<class ValueType, class Orientation, class IteratorTag, class Device>
struct matrix_temporary_type;

/// For the creation of temporary vectors in the assignment of proxies
template <class E>
struct vector_temporary{
	typedef typename vector_temporary_type<
		typename E::value_type,
		typename boost::mpl::eval_if<
			typename boost::is_base_of<vector_expression<E,typename E::device_category>,E>::type,
			boost::range_iterator<E>,
			major_iterator<E>
		>::type::iterator_category,
		typename E::device_category
	>::type type;
};

/// For the creation of temporary matrix in the assignment of proxies
template <class E>
struct matrix_temporary{
	typedef typename matrix_temporary_type<
		typename E::value_type,
		typename E::orientation,
		typename boost::mpl::eval_if<
			typename boost::is_base_of<vector_expression<E,typename E::device_category>,E>::type,
			boost::range_iterator<E>,
			major_iterator<E>
		>::type::iterator_category,
		typename E::device_category
	>::type type;
};

/// For the creation of transposed temporary matrix in the assignment of proxies
template <class E>
struct transposed_matrix_temporary{
	typedef typename matrix_temporary_type<
		typename E::value_type,
		typename E::orientation::transposed_orientation,
		typename boost::mpl::eval_if<
			typename boost::is_base_of<vector_expression<E,typename E::device_category>,E>::type,
			boost::range_iterator<E>,
			major_iterator<E>
		>::type::iterator_category,
		typename E::device_category
	>::type type;
};
	
}

#endif