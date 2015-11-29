/*!
 * 
 *
 * \brief       Operations and expression templates for expressions only involving vectors
 * 
 * 
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
#ifndef ABLAS_VECTOR_EXPRESSION_HPP
#define ABLAS_VECTOR_EXPRESSION_HPP

#include <boost/type_traits/is_convertible.hpp> 
#include <boost/utility/enable_if.hpp>

#include "assignment.hpp"
#include "detail/iterator.hpp"

namespace aBLAS{

/// \brief Vector expression being asynchronously evaluated
///
/// normally expressions of the form x=f(y)+g(z) are evaluated 
/// sequenially, e.g. first x=f(y) is evaluated and then x+=g(z).
/// if evaluation of either f(y) or g(z) is expensive, we can add further
/// parallelization using the async call:
/// x=async(f(y))+async(g(z))
/// which performs evaluation of the form
/// t1=f(y), t2= g(z), z=t1+t2;
/// where t1 and t2 can then be computed in parallel.
template<class E, class Device>
typename vector_temporary<E>::type async(vector_expression<E,Device> const& e){
	return typename vector_temporary<E>::type(e);
}
	
/// \brief Vector expression representing a constant valued vector.
template<class T, class Device>
class scalar_vector:public vector_expression<scalar_vector<T,Device>,Device > {
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef const T& const_reference;
	typedef const_reference reference;

	typedef std::size_t index_type;
	typedef scalar_vector const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename boost::mpl::if_<
		boost::is_same<Device,cpu_tag>,
		elementwise_tag,
		blockwise_tag
	>::type evaluation_category;
	typedef Device device_category;

	// Construction and destruction
	scalar_vector()
	:m_size(0), m_value() {}
	explicit scalar_vector(size_type size, value_type value)
	:m_size(size), m_value(value) {}
	scalar_vector(const scalar_vector& v)
	:m_size(v.m_size), m_value(v.m_value) {}

	// Accessors
	size_type size() const {
		return m_size;
	}
	
	std::vector<scheduling::dependency_node*> dependencies()const{
		return std::vector<scheduling::dependency_node*>();
	}

	// Element access
	const_reference operator()(index_type /*i*/) const {
		return m_value;
	}

	const_reference operator [](index_type /*i*/) const {
		return m_value;
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX,cpu_tag>& x, value_type alpha = value_type(1) )const{
		typename VecX::closure_type x_closure(x());
		value_type t = alpha * m_value;
		system::scheduler().spawn([t, x_closure](){
			kernels::assign<scalar_assign>(x_closure,t);
		},x().dependencies());
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_category>& x, value_type alpha = value_type(1) )const{
		x() += alpha * m_value;
	}

public:
	typedef constant_iterator<T> iterator;
	typedef constant_iterator<T> const_iterator;

	const_iterator begin() const {
		return const_iterator(0,m_value);
	}
	const_iterator end() const {
		return const_iterator(m_size,m_value);
	}

private:
	size_type m_size;
	value_type m_value;
};

///\brief Creates a vector having a constant value.
///
///@param alpha the value which is repeated
///@param elements the size of the resulting vector
template<class Device, class T>
scalar_vector<T,Device> repeat(T alpha, std::size_t elements){
	return scalar_vector<T,Device>(elements,alpha);
}
	
///\brief Implements multiplications of a vector by a alpha
template<class E>
class vector_scalar_multiply: public vector_expression<vector_scalar_multiply <E>, typename E::device_category > {
public:
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;
	typedef typename E::value_type value_type;
	typedef value_type const_reference;
	typedef value_type reference;

	typedef typename E::index_type index_type;

	typedef vector_scalar_multiply const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_category device_category;
private:
	expression_closure_type m_expression;
	value_type m_alpha;
public:

	// Construction and destruction
	// May be used as mutable expression.
	vector_scalar_multiply(expression_closure_type const& e, value_type alpha):
		m_expression(e), m_alpha(alpha) {}

	// Accessors
	size_type size() const {
		return m_expression.size();
	}

	auto dependencies()const -> decltype(this->m_expression.dependencies()){
		return m_expression.dependencies();
	}
	// Expression accessors
	expression_closure_type const &expression() const {
		return m_expression;
	}

	// Element access
	const_reference operator()(index_type i) const {
		return m_alpha * m_expression(i);
	}

	const_reference operator[](index_type i) const {
		return m_alpha * m_expression(i);
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_category>& x, value_type alpha = value_type(1) )const{
		m_expression.assign_to(x,alpha*m_alpha);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_category>& x, value_type alpha = value_type(1) )const{
		m_expression.plus_assign_to(x,alpha*m_alpha);
	}
	
	//iterators
	typedef transform_iterator<typename E::const_iterator,scalar_multiply1<value_type, value_type> > const_iterator;
	typedef const_iterator iterator;
	
	const_iterator begin() const {
		return const_iterator(m_expression.begin(),scalar_multiply1<value_type, value_type>(m_alpha));
	}
	const_iterator end() const {
		return const_iterator(m_expression.end(),scalar_multiply1<value_type, value_type>(m_alpha));
	}
};


template<class T, class E, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type >,
        vector_scalar_multiply<E>
>::type
operator* (vector_expression<E, Device> const& e, T alpha){
	typedef typename E::value_type value_type;
	return vector_scalar_multiply<E>(e(), value_type(alpha));
}
template<class T, class E, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type >,
        vector_scalar_multiply<E>
>::type
operator* (T alpha, vector_expression<E, Device> const& e){
	typedef typename E::value_type value_type;
	return vector_scalar_multiply<E>(e(), value_type(alpha));//explicit cast prevents warning, alternative would be to template vector_scalar_multiply on T as well
}

template<class E, class Device>
vector_scalar_multiply<E> operator-(vector_expression<E, Device> const& e){
	typedef typename E::value_type value_type;
	return vector_scalar_multiply<E>(e(), value_type(-1));//explicit cast prevents warning, alternative would be to template vector_scalar_multiply on T as well
}

template<class E1, class E2>
class vector_addition: public vector_expression<vector_addition<E1,E2>, typename E1::device_category > {
private:
	typedef scalar_binary_plus<
		typename E1::value_type,
		typename E2::value_type
	> functor_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const * const_pointer;
	typedef value_type const*  pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_closure_type expression_closure1_type;
	typedef typename E2::const_closure_type expression_closure2_type;
	
	typedef vector_addition<E1,E2> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;
	typedef typename E1::device_category device_category;

	// Construction and destruction
	explicit vector_addition (
		expression_closure1_type e1, 
		expression_closure2_type e2
	):m_lhs(e1),m_rhs(e2){
		ABLAS_SIZE_CHECK(e1.size() == e2.size());
	}

	// Accessors
	size_type size() const {
		return m_lhs.size();
	}

	std::vector<scheduling::dependency_node*> dependencies()const{
		return gather_dependencies(m_lhs.dependencies(),m_rhs.dependencies());
	}
	// Expression accessors
	expression_closure1_type const& expression1() const {
		return m_lhs;
	}
	expression_closure2_type const& expression2() const {
		return m_rhs;
	}

	// Element access
	const_reference operator() (index_type i) const {
		ABLAS_SIZE_CHECK(i < size());
		return m_lhs(i) + m_rhs(i);
	}

	const_reference operator[] (index_type i) const {
		ABLAS_SIZE_CHECK(i < size());
		return m_lhs(i) + m_rhs(i);
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_category>& x, value_type alpha = value_type(1) )const{
		assign(x, m_lhs,alpha);
		plus_assign(x, m_rhs,alpha);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_category>& x, value_type alpha = value_type(1) )const{
		plus_assign(x, m_lhs,alpha);
		plus_assign(x, m_rhs,alpha);
	}

	// Iterator types
	typedef binary_transform_iterator<
		typename E1::const_iterator,
		typename E2::const_iterator,
		functor_type
	> const_iterator;
	typedef const_iterator iterator;

	const_iterator begin () const {
		return const_iterator(functor_type(),
			m_lhs.begin(),m_lhs.end(),
			m_rhs.begin(),m_rhs.end()
		);
	}
	const_iterator end() const {
		return const_iterator(functor_type(),
			m_lhs.end(),m_lhs.end(),
			m_rhs.end(),m_rhs.end()
		);
	}

private:
	expression_closure1_type m_lhs;
	expression_closure2_type m_rhs;
};

///\brief Adds two vectors
template<class E1, class E2, class Device>
vector_addition<E1, E2 > operator+ (
	vector_expression<E1,  Device> const& e1,
	vector_expression<E2,  Device> const& e2
){
	return vector_addition<E1, E2>(e1(),e2());
}
///\brief Subtracts two vectors
template<class E1, class E2, class Device>
vector_addition<E1, vector_scalar_multiply<E2> > operator- (
	vector_expression<E1,  Device> const& e1,
	vector_expression<E2,  Device> const& e2
){
	return vector_addition<E1, vector_scalar_multiply<E2> >(e1(),-e2());
}

///\brief Adds a vector plus a scalr which is interpreted as a constant vector
template<class E, class T, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type>, 
	vector_addition<E, scalar_vector<T,Device> >
>::type operator+ (vector_expression<E,  Device> const& e, T t){
	return e + scalar_vector<T,Device>(e().size(),t);
}

///\brief Adds a vector plus a alpha which is interpreted as a constant vector
template<class T, class E, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type>,
	vector_addition<E, scalar_vector<T,Device> >
>::type operator+ (T t, vector_expression<E,  Device> const& e){
	return e + scalar_vector<T,Device>(e().size(),t);
}

///\brief Subtracts a alpha which is interpreted as a constant vector from a vector.
template<class E, class T, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type> ,
	vector_addition<E, vector_scalar_multiply<scalar_vector<T,Device> > >
>::type operator- (vector_expression<E,  Device> const& e, T t){
	return e - scalar_vector<T,Device>(e().size(),t);
}

///\brief Subtracts a vector from a alpha which is interpreted as a constant vector
template<class E, class T, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type>,
	vector_addition<scalar_vector<T,Device>, vector_scalar_multiply<E> >
>::type operator- (T t, vector_expression<E,  Device> const& e){
	return scalar_vector<T,Device>(e().size(),t) - e;
}

}

#endif
