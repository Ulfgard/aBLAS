/*!
 * 
 *
 * \brief       Matrix proxy classes.
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

#ifndef ABLAS_MATRIX_PROXY_HPP
#define ABLAS_MATRIX_PROXY_HPP

#include "assignment.hpp"
#include "detail/iterator.hpp"

namespace aBLAS{

/// \brief Matrix transpose.
template<class M>
class matrix_transpose: public matrix_expression<matrix_transpose<M>, typename M::device_category > {
public:
	typedef typename M::size_type size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::const_storage_type const_storage_type;
	typedef typename aBLAS::storage<M>::type storage_type;
	typedef typename M::index_type index_type;

	typedef typename closure<M>::type matrix_closure_type;
	typedef matrix_transpose<typename M::const_closure_type> const_closure_type;
	typedef matrix_transpose<M> closure_type;
	typedef typename M::orientation::transposed_orientation orientation;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;
	typedef typename M::device_category device_category;

	// Construction and destruction
	explicit matrix_transpose(matrix_closure_type const& m):
		m_expression(m) {}
	
	//conversion closure->const_closure
	template<class E>
	matrix_transpose(
		matrix_transpose<E> const& m,
		typename boost::disable_if<
			boost::mpl::or_<
				boost::is_same<matrix_transpose<E>,matrix_transpose>,
				boost::is_same<matrix_transpose<E>,matrix_closure_type>
			> 
		>::type* dummy = 0
	):m_expression(m.expression()) {}
		
	// Assignment
	//we implement it by using the identity A^T = B <=> A = B^T
	matrix_transpose& operator = (matrix_transpose const& m) {
		expression() = m.expression();
		return *this;
	}
	template<class E>
	matrix_transpose& operator = (matrix_expression<E,device_category> const& e) {
		expression() = matrix_transpose<E const>(e());
		return *this;
	}

	// Expression accessors
	matrix_closure_type const& expression() const{
		return m_expression;
	}
	matrix_closure_type expression(){
		return m_expression;
	}
	
	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return expression().size2();
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return expression().size1();
	}
	
	///\brief Returns the stride in memory between two rows.
	difference_type stride1()const{
		return expression().stride2();
	}
	///\brief Returns the stride in memory between two columns.
	difference_type stride2()const{
		return expression().stride1();
	}
	
	///\brief Returns the internal matrix storage
	storage_type storage()const{
		return expression().storage();
	}
	
	///\brief Returns the offset from the start of storage()
	difference_type offset()const{
		return expression().offset();
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
	reference operator()(index_type i, index_type j)const{
		return expression()(j, i);
	}

	typedef typename matrix_closure_type::const_column_iterator const_row_iterator;
	typedef typename matrix_closure_type::column_iterator row_iterator;
	typedef typename matrix_closure_type::const_row_iterator const_column_iterator;
	typedef typename matrix_closure_type::row_iterator column_iterator;

	//iterators
	row_iterator row_begin(index_type i) const {
		return expression().column_begin(i);
	}
	row_iterator row_end(index_type i) const {
		return expression().column_end(i);
	}
	column_iterator column_begin(index_type j) const {
		return expression().row_begin(j);
	}
	column_iterator column_end(index_type j) const {
		return expression().row_end(j);
	}
	
	void clear(){
		expression().clear();
	}
	
private:
	matrix_closure_type m_expression;
};


// (trans m) [i] [j] = m [j] [i]
template<class M, class Device>
matrix_transpose<typename M::const_closure_type>
trans(matrix_expression<M, Device> const& m) {
	return matrix_transpose<typename M::const_closure_type>(m());
}
template<class M, class Device>
temporary_proxy< matrix_transpose<M> > trans(matrix_expression<M, Device>& m) {
	return matrix_transpose<M>(m());
}

template<class M>
temporary_proxy< matrix_transpose<M> > trans(temporary_proxy<M> m) {
	return trans(static_cast<M&>(m));
}

template<class M>
class matrix_row: public vector_expression<matrix_row<M>, typename M::device_category > {
public:
	typedef M matrix_type;
	typedef std::size_t size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;

	typedef typename aBLAS::storage<M>::type const_storage_type;
	typedef const_storage_type storage_type;
	typedef typename M::index_type index_type;

	typedef typename closure<M>::type matrix_closure_type;
	typedef matrix_row<typename M::const_closure_type> const_closure_type;
	typedef matrix_row<M> closure_type;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;
	typedef typename M::device_category device_category;

	// Construction and destruction
	matrix_row(matrix_closure_type const& expression, index_type i):m_expression(expression), m_i(i) {
		ABLAS_SIZE_CHECK (i < expression.size1());
	}
	
	template<class E>
	matrix_row(matrix_row<E> const& other)
	:m_expression(other.expression()),m_i(other.index()){}
		
	// Assignment
	template<class E>
	matrix_row& operator = (vector_expression<E, device_category> const& e) {
		typedef typename vector_temporary<M>::type Temporary;
		system::scheduler().create_closure(
			Temporary(e.size()),
			[this,&e](Temporary& temporary){
				assign(temporary,e);
				assign(*this,temporary);
			}
		);
	}
	matrix_row& operator = (matrix_row const& e) {
		typedef typename vector_temporary<M>::type Temporary;
		system::scheduler().create_closure(
			Temporary(e.size()),
			[this,&e](Temporary& temporary){
				assign(temporary,e);
				assign(*this,temporary);
			}
		);
	}
	
	matrix_closure_type const& expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	index_type index() const {
		return m_i;
	}
	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the size of the vector
	size_type size() const {
		return expression().size2();
	}
	
	///\brief Returns the stride in memory between two elements
	difference_type stride()const{
		return expression().stride2();
	}
	
	///\brief Returns the offset from the start of storage()
	difference_type offset()const{
		return expression().offset()+index()*expression().stride1();
	}
	
	///\brief Returns the internal matrix storage
	storage_type storage()const{
		return expression().storage();
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
	reference operator()(index_type j) const {
		return m_expression(m_i, j);
	}
	reference operator [](index_type j) const {
		return (*this)(j);
	}
	
	// Iterator types
	typedef typename M::const_row_iterator const_iterator;
	typedef typename row_iterator<M>::type iterator;

	iterator begin()const{
		return expression().row_begin(m_i);
	}
	iterator end()const{
		return expression().row_end(m_i);
	}
	
	void clear(){
		closure_type closure(*this);
		system::scheduler().spawn([closure]()mutable{
			kernels::assign<scalar_assign>(closure,value_type/* zero */());
		},dependencies());
	}
	
private:
	matrix_closure_type m_expression;
	size_type m_i;
};

// Projections
template<class M, class Device>
temporary_proxy< matrix_row<M> > row(matrix_expression<M, Device>& expression, typename M::index_type i) {
	return matrix_row<M> (expression(), i);
}
template<class M, class Device>
matrix_row<typename M::const_closure_type>
row(matrix_expression<M, Device> const& expression, typename M::index_type i) {
	return matrix_row<typename M::const_closure_type> (expression(), i);
}

template<class M>
temporary_proxy<matrix_row<M> > row(temporary_proxy<M> expression, typename M::index_type i) {
	return row(static_cast<M&>(expression), i);
}

template<class M>
class matrix_column: public vector_expression<matrix_column<M>, typename M::device_category > {
public:
	typedef M matrix_type;
	typedef std::size_t size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;

	typedef typename M::const_storage_type const_storage_type;
	typedef typename storage<M>::type storage_type;
	typedef typename M::index_type index_type;

	typedef typename closure<M>::type matrix_closure_type;
	typedef matrix_column<typename M::const_closure_type> const_closure_type;
	typedef matrix_column<M> closure_type;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;
	typedef typename M::device_category device_category;

	// Construction and destruction
	matrix_column(matrix_closure_type const& expression, index_type j)
	:m_expression(expression), m_j(j) {
		ABLAS_SIZE_CHECK (j < expression.size2());
	}
	
	template<class E>
	matrix_column(matrix_column<E> const& other)
	:m_expression(other.expression()),m_j(other.index()){}
		
	// Assignment
	
	template<class E>
	matrix_column& operator = (vector_expression<E,device_category> const& e) {
		typedef typename vector_temporary<M>::type Temporary;
		system::scheduler().create_closure(
			Temporary(e().size()),
			[this,&e](Temporary& temporary){
				assign(temporary,e);
				assign(*this,temporary);
			}
		);
	}
	matrix_column& operator = (matrix_column const& e) {
		typedef typename vector_temporary<M>::type Temporary;
		system::scheduler().create_closure(
			Temporary(e().size()),
			[this,&e](Temporary& temporary){
				assign(temporary,e);
				assign(*this,temporary);
			}
		);
	}
	
	matrix_closure_type const& expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	index_type index() const {
		return m_j;
	}
	
	///\brief Returns the size of the vector
	size_type size() const {
		return expression().size1();
	}

	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the stride in memory between two elements
	difference_type stride()const{
		return expression().stride1();
	}
	
	///\brief Returns the internal matrix storage
	storage_type storage()const{
		return expression().storage();
	}
	
	///\brief Returns the offset from the start of storage()
	difference_type offset()const{
		return expression().offset()+index()*expression().stride2();
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
	reference operator()(index_type i) const {
		return m_expression(i,m_j);
	}
	reference operator [](index_type i) const {
		return (*this)(i);
	}
	

	// Iterator types
	typedef typename M::const_column_iterator const_iterator;
	typedef typename column_iterator<M>::type iterator;

	iterator begin()const{
		return expression().column_begin(m_j);
	}
	iterator end()const{
		return expression().column_end(m_j);
	}
	
	void clear(){
		closure_type closure(*this);
		system::scheduler().spawn([closure]()mutable{
			kernels::assign<scalar_assign>(closure,value_type/* zero */());
		},dependencies());
	}

private:
	matrix_closure_type m_expression;
	size_type m_j;
};

// Projections
template<class M, class Device>
temporary_proxy<matrix_column<M> > column(matrix_expression<M, Device>& expression, typename M::index_type j) {
	return matrix_column<M> (expression(), j);
}
template<class M, class Device>
matrix_column<typename M::const_closure_type>
column(matrix_expression<M, Device> const& expression, typename M::index_type j) {
	return matrix_column<typename M::const_closure_type> (expression(), j);
}

template<class M>
temporary_proxy<matrix_column<M> > column(temporary_proxy<M> expression, typename M::index_type j) {
	return column(static_cast<M&>(expression), j);
}

// Matrix based range class
template<class M>
class matrix_range:public matrix_expression<matrix_range<M>, typename M::device_category > {
public:
	typedef M matrix_type;
	typedef std::size_t size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;

	typedef typename M::index_type index_type;
	typedef typename M::const_storage_type const_storage_type;
	typedef typename storage<M>::type storage_type;

	typedef typename closure<M>::type matrix_closure_type;
	typedef matrix_range<typename M::const_closure_type> const_closure_type;
	typedef matrix_range<M> closure_type;
	typedef typename M::storage_category storage_category;
	typedef elementwise_tag evaluation_category;
	typedef typename M::orientation orientation;
	typedef typename M::device_category device_category;

	// Construction and destruction

	matrix_range(matrix_closure_type expression, range const&r1, range const&r2)
	:m_expression(expression), m_range1(r1), m_range2(r2) {
		ABLAS_SIZE_CHECK(r1.start() <= expression.size1());
		ABLAS_SIZE_CHECK(r1.start() +r1.size() <= expression.size1());
		ABLAS_SIZE_CHECK(r2.start() <= expression.size2());
		ABLAS_SIZE_CHECK(r2.start() +r2.size() <= expression.size2());
	}
	
	//conversion closure->const_closure
	template<class E>
	matrix_range(
		matrix_range<E> const& other,
		typename boost::disable_if<
			boost::is_same<E,matrix_range>
		>::type* dummy = 0
	):m_expression(other.expression())
	, m_range1(other.range1())
	, m_range2(other.range2()){}
		
	// Accessors
	size_type start1() const {
		return m_range1.start();
	}
	size_type start2() const {
		return m_range2.start();
	}

	matrix_closure_type expression() const {
		return m_expression;
	}
	matrix_closure_type& expression() {
		return m_expression;
	}
	
	range const& range1() const{
		return m_range1;
	}
	range const& range2() const{
		return m_range2;
	}
	
	// ---------
	// Dense Low level interface
	// ---------
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return m_range1.size();
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_range2.size();
	}
	
	///\brief Returns the stride in memory between two rows.
	difference_type stride1()const{
		return expression().stride1();
	}
	///\brief Returns the stride in memory between two columns.
	difference_type stride2()const{
		return expression().stride2();
	}
	
	///\brief Returns the offset from the start of storage()
	difference_type offset()const{
		return expression().offset()+start1()*stride1()+start2()*stride2();
	}
	
	///\brief Returns the internal matrix storage
	storage_type storage()const{
		return expression().storage();
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
	reference operator()(index_type i, index_type j)const{
		return m_expression(m_range1(i), m_range2(j));
	}

	// Assignment
	
	matrix_range& operator = (matrix_range const& e) {
		return assign(*this, typename matrix_temporary<matrix_range>::type(e));
	}
	template<class E>
	matrix_range& operator = (matrix_expression<E, device_category> const& e) {
		return assign(*this, typename matrix_temporary<E>::type(e));
	}

	// Iterator types
	typedef subrange_iterator<typename M::row_iterator> row_iterator;
	typedef subrange_iterator<typename M::column_iterator> column_iterator;
	typedef subrange_iterator<typename M::const_row_iterator> const_row_iterator;
	typedef subrange_iterator<typename M::const_column_iterator> const_column_iterator;

	// Element lookup
	row_iterator row_begin(index_type i) const {
		return row_iterator(
			expression().row_begin(i+start1()),expression().row_end(i+start1()),
			start2(),start2()
		);
	}
	row_iterator row_end(index_type i) const {
		return row_iterator(
			expression().row_begin(i+start1()),expression().row_end(i+start1()),
			start2()+size2(),start2()
		);
	}
	column_iterator column_begin(index_type j) const {
		return column_iterator(
			expression().column_begin(j+start2()),expression().column_end(j+start2()),
			start1(),start1()
		);
	}
	column_iterator column_end(index_type j) const {
		return column_iterator(
			expression().column_begin(j+start2()),expression().column_end(j+start2()),
			start1()+size1(),start1()
		);
	}
	
	void clear(){
		closure_type closure(*this);
		system::scheduler().spawn([closure]()mutable{
			kernels::assign<scalar_assign>(closure,value_type/* zero */());
		},dependencies());
	}

private:
	matrix_closure_type m_expression;
	range m_range1;
	range m_range2;
};

// Simple Subranges
template<class M, class Device>
temporary_proxy< matrix_range<M> > subrange(
	matrix_expression<M, Device>& expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
) {
	ABLAS_RANGE_CHECK(start1 <= stop1);
	ABLAS_RANGE_CHECK(start2 <= stop2);
	ABLAS_SIZE_CHECK(stop1 <= expression().size1());
	ABLAS_SIZE_CHECK(stop2 <= expression().size2());
	return matrix_range<M> (expression(), range(start1, stop1), range(start2, stop2));
}
template<class M, class Device>
matrix_range<typename M::const_closure_type>
subrange(
	matrix_expression<M, Device> const& expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
) {
	ABLAS_RANGE_CHECK(start1 <= stop1);
	ABLAS_RANGE_CHECK(start2 <= stop2);
	ABLAS_SIZE_CHECK(stop1 <= expression().size1());
	ABLAS_SIZE_CHECK(stop2 <= expression().size2());
	return matrix_range<typename M::const_closure_type> (expression(), range(start1, stop1), range(start2, stop2));
}

template<class M>
temporary_proxy< matrix_range<M> > subrange(
	temporary_proxy<M> expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
) {
	return subrange(static_cast<M&>(expression),start1,stop1,start2,stop2);
}

//obtaining a set of consecutive rows of a matrix
template<class M, class Device>
temporary_proxy< matrix_range<M> > rows(
	matrix_expression<M, Device>& expression, 
	std::size_t start, std::size_t stop
) {
	ABLAS_RANGE_CHECK(start <= stop);
	ABLAS_SIZE_CHECK(stop <= expression().size1());
	return matrix_range<M> (expression(), range(start, stop), range(0,expression().size2()));
}
template<class M, class Device>
matrix_range<typename M::const_closure_type>
rows(
	matrix_expression<M, Device> const& expression, 
	std::size_t start, std::size_t stop
) {
	ABLAS_RANGE_CHECK(start <= stop);
	ABLAS_SIZE_CHECK(stop <= expression().size1());
	return matrix_range<typename M::const_closure_type> (expression(), range(start, stop), range(0,expression().size2()));
}

template<class M>
temporary_proxy< matrix_range<M> > rows(
	temporary_proxy<M> expression, 
	std::size_t start, std::size_t stop
) {
	return subrange(static_cast<M&>(expression),start,stop);
}

//obtaining a set of consecutive columns of a matrix
template<class M, class Device>
temporary_proxy< matrix_range<M> > columns(
	matrix_expression<M, Device>& expression, 
	std::size_t start, std::size_t stop
) {
	ABLAS_RANGE_CHECK(start <= stop);
	ABLAS_SIZE_CHECK(stop <= expression().size2());
	return matrix_range<M> (expression(), range(0,expression().size1()), range(start, stop));
}
template<class M, class Device>
matrix_range<typename M::const_closure_type>
columns(
	matrix_expression<M, Device> const& expression, 
	std::size_t start, std::size_t stop
) {
	ABLAS_RANGE_CHECK(start <= stop);
	ABLAS_SIZE_CHECK(stop <= expression().size2());
	return matrix_range<typename M::const_closure_type> (expression(), range(0,expression().size1()), range(start, stop));
}

template<class M>
temporary_proxy< matrix_range<M> > columns(
	temporary_proxy<M> expression, 
	std::size_t start, std::size_t stop
) {
	return subrange(static_cast<M&>(expression),start,stop);
}

}
#endif
