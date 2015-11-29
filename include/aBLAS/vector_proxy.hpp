/*!
 * 
 *
 * \brief       Vector proxy classes.
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
#ifndef ABLAS_VECTOR_PROXY_HPP
#define ABLAS_VECTOR_PROXY_HPP

#include "assignment.hpp"
#include "detail/iterator.hpp"

namespace aBLAS{

/** \brief A vector referencing a continuous subvector of elements of vector \c v containing all elements specified by \c range.
 *
 * A vector range can be used as a normal vector in any expression.
 * If the specified range falls outside that of the index range of the vector, then
 * the \c vector_range is not a well formed \c vector_expression and access to an
 * element outside of index range of the vector is \b undefined.
 *
 * \tparam V the type of vector referenced (for exaboost::mple \c vector<double>)
 */
template<class V>
class vector_range:public vector_expression<vector_range<V>, typename V::device_category >{
public:
	typedef typename closure<V>::type vector_closure_type;

	typedef typename V::size_type size_type;
	typedef typename V::difference_type difference_type;
	typedef typename V::value_type value_type;
	typedef typename V::scalar_type scalar_type;
	typedef typename V::const_reference const_reference;
	typedef typename reference<V>::type reference;

	typedef typename aBLAS::storage<V>::type const_storage_type;
	typedef const_storage_type storage_type;
	typedef typename V::index_type index_type;
	
	typedef vector_range<typename V::const_closure_type> const_closure_type;
	typedef vector_range<V> closure_type;
	typedef typename V::storage_category storage_category;
	typedef elementwise_tag evaluation_category;
	typedef typename V::device_category device_category;

	// Construction and destruction
	vector_range(vector_closure_type const& data, range const& r):
		m_expression(data), m_range(r){
		RANGE_CHECK(start() <= m_expression.size());
		RANGE_CHECK(start() + size() <= m_expression.size());
	}
	
	//non-const-> const conversion
	template<class E>
	vector_range(
		vector_range<E,device_category> const& other,
		typename boost::disable_if<
			boost::is_same<E,vector_range>
		>::type* dummy = 0
	):m_expression(other.expression())
	, m_range(other.range()){}
		
	// Assignment operators 
	vector_range& operator = (vector_range const& e){
		typedef typename vector_temporary<V>::type Temporary;
		system::scheduler().create_closure(
			Temporary(e.size()),
			[this,&e](Temporary& temporary){
				assign(temporary,e);
				assign(*this,temporary);
			}
		);
	}

	template<class E>
	vector_range& operator = (vector_expression<E, device_type> const& e){
		typedef typename vector_temporary<V>::type Temporary;
		system::scheduler().create_closure(
			Temporary(e.size()),
			[this,&e](Temporary& temporary){
				assign(temporary,e);
				assign(*this,temporary);
			}
		);
	}
	
	// ---------
	// Internal Accessors
	// ---------
	
	size_type start() const{
		return m_range.start();
	}
	
	vector_closure_type const& expression() const{
		return m_expression;
	}
	vector_closure_type& expression(){
		return m_expression;
	}
	
	blas::range const& range()const{
		return m_range;
	}
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_range.size();
	}

	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the pointer to the beginning of the vector storage
	///
	/// Low-level access to the vectors internals. Elements storage()[offset()+i*stride()] for i=1,...,size()-1 are valid
	storage_type storage()const{
		return expression().storage();
	}
	
	///\brief Returns the stride between the elements in storage()
	///
	/// In general elements of dense storage entities are spaced like storage()[offset()+i*stride()] for i=1,...,size()-1
	difference_type stride()const{
		return expression().stride();
	}
	
	///\brief Returns the offset from the start of storage()
	difference_type offset()const{
		return expression().offset()+start()*stride();
	}
	
	// ---------
	// Async Interface
	// ---------
	
	/// \brief Returns true if this vector does not wait for operations to complete
	bool is_ready(){
		return expression().is_ready();
	}
	
	/// \brief Blocks this thread until all kernels are computed.
	///
	/// Be aware that when other threads enqueue kernels in parallel while wait()
	/// is called, wait() can not guarantee that is_ready() is true after wait() returns.
	/// Thus, writing access to the vector after wait() has been called is undefined when there
	/// are other threads using the vector.
	void wait(){
		expression().wait();
	}
	
	///\brief Returns the dependices of this vector.
	scheduling::dependency_node& dependencies() const{
		return expression().dependencies();
	}
	
	// ---------
	// High level interface
	// ---------

	// Element access
	reference operator()(index_type i) const{
		return m_expression(m_range(i));
	}

	typedef subrange_iterator< typename vector_closure_type::iterator> iterator;
	typedef subrange_iterator< typename vector_closure_type::const_iterator> const_iterator;

	iterator begin() const{
		return iterator(
		        m_expression.begin(),m_expression.end(),
		        start(),start()
		);
	}
	iterator end() const{
		return iterator(
		        m_expression.begin(),m_expression.end(),
		        start()+size(),start()
		);
	}
	
	void clear(){
		closure_type closure(*this);
		system::scheduler().spawn([closure]()mutable{
			kernels::assign<scalar_assign>(closure,value_type/* zero */());
		},dependencies());
	}
private:
	vector_closure_type m_expression;
	aBLAS::range m_range;
};

// ------------------
// Subranges
// ------------------

/** \brief Return a \c vector_range on a specified vector, a start and stop index.
 * Return a \c vector_range on a specified vector, a start and stop index. The resulting \c vector_range can be manipulated like a normal vector.
 * If the specified range falls outside that of of the index range of the vector, then the resulting \c vector_range is not a well formed
 * Vector Expression and access to an element outside of index range of the vector is \b undefined.
 */
template<class V, class Device>
temporary_proxy<vector_range<V> > subrange(vector_expression<V, Device>& data, typename V::size_type start, typename V::size_type stop){
	return vector_range<V> (data(), range(start, stop));
}

/** \brief Return a \c const \c vector_range on a specified vector, a start and stop index.
 * Return a \c const \c vector_range on a specified vector, a start and stop index. The resulting \c const \c vector_range can be manipulated like a normal vector.
 *If the specified range falls outside that of of the index range of the vector, then the resulting \c vector_range is not a well formed
 * Vector Expression and access to an element outside of index range of the vector is \b undefined.
 */
template<class V, class Device>
auto subrange(
	vector_expression<V, Device> const& data,
	typename V::size_type start,
	typename V::size_type stop
){
	return vector_range<typename const_expression<V>::type> (data(), range(start, stop));
}

template<class V>
temporary_proxy<vector_range<V> > subrange(temporary_proxy<V> data, typename V::size_type start, typename V::size_type stop){
	return subrange(static_cast<V&>(data), start, stop);
}


}

#endif
