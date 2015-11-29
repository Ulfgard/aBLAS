/*!
 * 
 *
 * \brief       Implements the dense matrix container class.
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
#ifndef ABLAS_MATRIX_HPP
#define ABLAS_MATRIX_HPP

#include <boost/container/vector.hpp>
#include <memory>

#include "assignment.hpp"
#include "detail/iterator.hpp"

namespace aBLAS { namespace detail{
	
template<class T>
struct dense_matrix_state{
	typedef boost::container::vector<T> storage_type;//better handling of std::matrix<bool>...
	typedef storage_type const const_storage_type;
	typedef typename storage_type::size_type size_type;
	typedef typename storage_type::value_type value_type;
	storage_type data;
	mutable scheduling::dependency_node dependencies;
	size_type size1;
	size_type size2;
	
	dense_matrix_state():size1(0),size2(0){}
	dense_matrix_state(size_type size1, size_type size2):data(size1*size2), size1(size1),size2(size2){}
	dense_matrix_state(size_type size1, size_type size2, value_type init):data(size1* size2, init), size1(size1),size2(size2){}
};
	
template<class SharedState, class O, class IsNonConstReference =  boost::mpl::false_>
class dense_matrix_base{
protected:
	void set_state(SharedState* state){
		m_internals = state;
	}
	
	template<class,class, class> friend class dense_matrix_base;
public:
	typedef typename boost::mpl::if_<
		boost::is_const<SharedState>,
	        typename SharedState::const_storage_type,
	        typename SharedState::storage_type
	>::type storage_type;
	typedef typename SharedState::const_storage_type const_storage_type;
	typedef typename storage_type::size_type size_type;
	typedef typename storage_type::difference_type difference_type;
	typedef typename storage_type::value_type value_type;
	typedef typename storage_type::const_reference const_reference;
	typedef typename boost::mpl::if_<
		boost::is_const<SharedState>,
	        typename storage_type::const_reference ,
		typename storage_type::reference 
	>::type reference;
	
	typedef std::size_t index_type;

	typedef dense_tag storage_category;
	typedef elementwise_tag evaluation_category;
	typedef cpu_tag device_category;
	typedef O orientation;

	// Construction
	dense_matrix_base(){}
		
	//allows conversion non-const->const
	template<class S, class R>
	dense_matrix_base(dense_matrix_base<S,O,R> const& state):m_internals(state.m_internals){}
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return m_internals->size1;
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_internals->size2;
	}
	
	/// \brief Return true if the matrix is empty
	/// \return \c true if empty, \c false otherwise
	bool empty() const {
		return storage().empty();
	}
	
	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the internal matrix storage
	///
	/// In general elements of dense storage entities are spaced like 
	/// storage()[offset()+i*stride1()+j*stride2()] for i=0,...,size1()-1, j=0,...,size2()-1
	storage_type& storage(){
		return m_internals->data;
	}
	
	///\brief Returns the internal matrix storage
	///
	/// In general elements of dense storage entities are spaced like 
	/// storage()[offset()+i*stride1()+j*stride2()] for i=0,...,size1()-1, j=0,...,size2()-1
	typename boost::mpl::if_<
		IsNonConstReference,
		storage_type,
		const_storage_type
	>::type& storage()const{
		return m_internals->data;
	}
	
	///\brief Returns the stride in memory between two rows.
	difference_type stride1()const{
		return orientation::stride1(size1(),size2());
	}
	///\brief Returns the stride in memory between two columns.
	difference_type stride2()const{
		return orientation::stride2(size1(),size2());
	}
	
	///\brief Returns the offset from the start of storage()
	difference_type offset()const{
		return 0;
	}
	
	// ---------
	// Async Interface
	// ---------
	
	/// \brief Returns true if this matrix does not wait for operations to complete
	bool is_ready()const{
		return !m_internals || m_internals->dependencies.is_ready();
	}
	
	/// \brief Blocks this thread until all kernels are computed.
	///
	/// Be aware that when other threads enqueue kernels in parallel while wait()
	/// is called, wait() can not guarantee that is_ready() is true after wait() returns.
	/// Thus, writing access to the matrix after wait() has been called is undefined when there
	/// are other threads using the matrix.
	void wait(){
		m_internals->dependencies.wait();
	}
	
	///\brief Returns the dependices of this matrix.
	scheduling::dependency_node& dependencies() const{
		return m_internals->dependencies;
	}
	
	// ---------
	// High level interface
	// ---------

	// Element access
	
	/// \brief Returns element (i,j) of the matrix
	/// \param i the row index
	/// \param j the column index
	typename boost::mpl::if_<
		IsNonConstReference,
		reference,
		const_reference
	>::type& operator()(index_type i, index_type j) const {
		return storage()[orientation::element(i, size1(), j, size2())];
	}
	/// \brief Returns element (i,j) of the matrix
	/// \param i the row index
	/// \param j the column index
	reference operator()(index_type i, index_type j) {
		return storage()[orientation::element(i, size1(), j, size2())];
	}

	//Iterators
private:
	typedef typename boost::mpl::if_<boost::is_const<storage_type>,
	        typename storage_type::const_iterator,
	        typename storage_type::iterator
	>::type storage_iterator;
public:
	typedef dense_storage_iterator<storage_iterator> row_iterator;
	typedef dense_storage_iterator<storage_iterator> column_iterator;
	typedef dense_storage_iterator<typename storage_type::const_iterator> const_row_iterator;
	typedef dense_storage_iterator<typename storage_type::const_iterator> const_column_iterator;
private:
	typedef typename boost::mpl::if_<
		IsNonConstReference,
		row_iterator,
		const_row_iterator
	>::type reference_const_row_iterator;
	typedef typename boost::mpl::if_<
		IsNonConstReference,
		column_iterator,
		const_column_iterator
	>::type reference_const_column_iterator;
public:

	reference_const_row_iterator row_begin(index_type i) const {
		return reference_const_row_iterator(storage().begin() + i*stride1(),0,stride2());
	}
	reference_const_row_iterator row_end(index_type i) const {
		return reference_const_row_iterator(storage().begin() + i*stride1()+stride2()*size2(),size2(),stride2());
	}
	row_iterator row_begin(index_type i){
		return row_iterator(storage().begin() + i*stride1(),0,stride2());
	}
	row_iterator row_end(index_type i){
		return row_iterator(storage().begin() + i*stride1()+stride2()*size2(),size2(),stride2());
	}
	
	reference_const_row_iterator column_begin(std::size_t j) const {
		return reference_const_column_iterator(storage().begin()+j*stride2(),0,stride1());
	}
	reference_const_column_iterator column_end(std::size_t j) const {
		return reference_const_column_iterator(storage().begin()+j*stride2()+ stride1()*size1(),size1(),stride1());
	}
	column_iterator column_begin(std::size_t j){
		return column_iterator(storage().begin()+j*stride2(),0,stride1());
	}
	column_iterator column_end(std::size_t j){
		return column_iterator(storage().begin()+j*stride2()+ stride1()*size1(),size1(),stride1());
	}

private:
	SharedState* m_internals;
};
	
} //namespace detail

/// \brief A dense matrix of values of type \c T.
///
/// For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
/// the \f$(i*n + j)\f$-th element of the container for row major orientation or the \f$ (i + j*m) \f$-th element of
/// the container for column major orientation. In a dense matrix all elements are represented in memory in a
/// contiguous chunk of memory by definition.
///
/// Orientation can also be specified, otherwise a \c row_major is used.
///
/// \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
/// \tparam O the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
template<class T, class O=row_major>
class matrix
	: public matrix_expression<matrix<T, O>,cpu_tag >
	, public detail::dense_matrix_base<detail::dense_matrix_state<T>, O> {
private:
	typedef typename detail::dense_matrix_base<detail::dense_matrix_state<T>,O> base;

	struct closure_type_base
	: public matrix_expression<closure_type_base, cpu_tag >,
	  public detail::dense_matrix_base<detail::dense_matrix_state<T>, O, boost::mpl::true_>
	{
		typedef typename matrix<T,O>::const_closure_type const_closure_type;
		typedef typename matrix<T,O>::closure_type closure_type;
		closure_type_base(matrix const& v):detail::dense_matrix_base<detail::dense_matrix_state<T>, O, boost::mpl::true_>(v){}
			
		// -------------------
		// Assignment operators
		// -------------------
			
		using detail::dense_matrix_base<detail::dense_matrix_state<T>, O, boost::mpl::true_>::operator();
		
		/// \brief Operator=
		matrix& operator = (closure_type_base const& v) {
			ABLAS_SIZE_CHECK(v.size() == this->size());
			assign(*this,v);
		}

		/// \brief Assign the result of a matrix_expression to the matrix
		/// \param v is a const reference to the matrix_expression
		/// \return a reference to the resulting matrix
		template<class V>
		matrix& operator = (matrix_expression<V, cpu_tag> const& v) {
			ABLAS_SIZE_CHECK(v.size() == this->size());
			assign(*this,v);
		}
		
		/// \brief Clear the matrix, i.e. set all values to the \c zero value.
		void clear() {
			//running via the scheduler is a lot of overhead for a simple operation
			//so if no kernels are in flight, we can just call the kernel directly
			//otherwise, we have to enqueue it.
			if(is_ready()){
				kernels::assign<scalar_assign>(*this, value_type/*zero*/());
			}else{
				closure_type_base closure(*this);
				system::scheduler().spawn([closure]()mutable{
					kernels::assign<scalar_assign>(closure, value_type/*zero*/());
				},this->dependencies());
			}
		}
	};
	
	struct const_closure_type_base
	: public matrix_expression<const_closure_type_base, cpu_tag >,
	  public detail::dense_matrix_base<detail::dense_matrix_state<T> const,O >
	{
		typedef typename matrix<T,O>::const_closure_type const_closure_type;
		typedef typename matrix<T,O>::const_closure_type closure_type;

		using detail::dense_matrix_base<detail::dense_matrix_state<T> const,O >::operator();
		
		const_closure_type_base(matrix const& v):detail::dense_matrix_base<detail::dense_matrix_state<T> const,O >(v){}
		//constructor for non-const->const copying
		const_closure_type_base(closure_type_base const& c)
		:detail::dense_matrix_base<detail::dense_matrix_state<T> const,O >(c){}
	};

public:
	typedef typename base::size_type size_type;
	typedef typename base::value_type value_type;
	typedef closure_type_base closure_type;
	typedef const_closure_type_base const_closure_type;
	using base::is_ready;
	using base::storage;
	using base::set_state;
	using base::size1;
	using base::size2;
	using base::operator();

	// Construction and destruction

	/// Default dense matrix constructor. Make a dense matrix of size (0,0)
	// Construction and destruction

	/// \brief Default Dense Matrix constructor of size (0,0)
	matrix():m_internals(new detail::dense_matrix_state<T>()) {
		set_state(m_internals.get());
	}

	/// \brief Constructor of a matrix with a predefined size
	/// \param size1 number of rows of the matrix
	/// \param size2 number of columns of the matrix
	matrix(size_type size1, size_type size2):m_internals(new detail::dense_matrix_state<T>(size1,size2)) {
		set_state(m_internals.get());
	}

	/// \brief Constructor of a matrix with a predefined size with all elements initialized to an initial value
	/// \param size1 number of rows of the matrix
	/// \param size2 number of columns of the matrix
	/// \param init value to assign to each element of the matrix
	matrix(size_type size1, size_type size2, value_type init):m_internals(new detail::dense_matrix_state<T>(size1,size2, init)) {
		set_state(m_internals.get());
	}

	/// \brief Copy-constructor of a matrix
	/// \param m is the matrix to be copied
	matrix(matrix const& m):m_internals(new detail::dense_matrix_state<T>(m.size1(), m.size2())) {
		set_state(m_internals.get());
		if(m.is_ready())//m has no kernels in flight, just copy
			storage() = m.storage();
		else
			assign(*this,m);//start assignment kernel
	}
		
	/// \brief Move Constructor
	///
	///Moving a matrix with active kernels is a well defined operation and guaranteed to work and non-blocking.
	matrix(matrix && m): m_internals(std::move(m.m_internals)){
		set_state(m_internals.get());
	}

	/// \brief Creates a matrix from a matrix_expression
	/// \param m the matrix_expression which values will be assigned to the matrix
	template<class M>
	matrix(matrix_expression<M, cpu_tag> const& m)
	:m_internals(new detail::dense_matrix_state<T>(m().size1(),m().size2())){
		set_state(m_internals.get());
		assign(*this, m);
	}
	
	~matrix(){
		//if this still owns memory and there are still kernels in flight, transfer ownership to the scheduler
		//this delays destruction until the last kernel using *this is finished
		if(!is_ready())
			system::scheduler().make_closure_variable(*this);
	}
	
	// -------------------
	// Assignment operators
	// -------------------
	
	/// \brief Operator=
	matrix& operator = (matrix const& m) {
		//if this matrix is not used, we do not need to create a copy
		if(is_ready()){
			m_internals->data.resize(m.size1()*m.size2());
			m_internals->size1 = m.size1;
			m_internals->size2 = m.size2;
			
			if(m.is_ready())//v has no kernels in flight, just copy
				storage() = m.storage();
			else
				assign(*this,m);//start assignment kernel
		}else{
			matrix temporary(m);//start assignment kernel in temporary
			swap(*this,temporary);//swap detail::dense_matrix_state<T>, destructor of the temporary will transfer ownership to the scheduler
		}
		return *this;
	}
	
	/// \brief Move Operator=
	matrix& operator = (matrix && m) {
		//if this matrix is used, we have to transfer ownership to the scheduler
		if(!is_ready())
			system::scheduler().make_closure_variable(*this);
		m_internals = std::move(m.m_internals);
		set_state(m_internals.get());
		return *this;
	}
	
	/// \brief Assign the result of a matrix_expression to the matrix
	///
	/// This operator assumes that the expressions are aliasing and thus always stores 
	/// the result of the expression in a temporary before copying
	/// \param mis a const reference to the matrix_expression
	/// \return a reference to the resulting matrix
	template<class M>
	matrix& operator = (matrix_expression<M, cpu_tag> const& m) {
		matrix temporary(m);
		swap(*this,temporary);
		return *this;
	}
	
	/// \brief Resizes the matrix
	///
	///The value of the elements after resize are undefined.
	///Be aware that resizing while kernels are in flight leads
	///to the creation of a new variable.
	void resize(size_type new_size1, size_type new_size2){
		if(new_size1 == size1() && new_size2 == size2())
			return;
		if(is_ready()){
			m_internals->data.resize(new_size1 * new_size1);
			m_internals->size1 = new_size1;
			m_internals->size2 = new_size2;
		}else{
			//there are still kernels using this matrix, so create a new variable and let the scheduler handle everything
			matrix temporary(new_size1, new_size2);
			swap(*this,temporary);
		}
	}
	
	/// \brief Swap the content of two matrices
	friend void swap(matrix& m1, matrix& m2) {
		m1.m_internals.swap(m2.m_internals);
		std::swap(static_cast<base&>(m1),static_cast<base&>(m2));
	}
	
	/// \brief Clear the matrix, i.e. set all values to the \c zero value.
	void clear() {
		//running via the scheduler is a lot of overhead for a simple operation
		//so if no kernels are in flight, we can just call the kernel directly
		//otherwise, we have to enqueue it.
		if(is_ready()){
			kernels::assign<scalar_assign>(*this, value_type/*zero*/());
		}else{
			closure_type closure(*this);
			system::scheduler().spawn([closure]()mutable{
				kernels::assign<scalar_assign>(closure, value_type/*zero*/());
			},this->dependencies());
		}
	}
private:
	std::unique_ptr<detail::dense_matrix_state<T> > m_internals;
};
template<class T, class L>
struct matrix_temporary_type<T,L,dense_random_access_iterator_tag, cpu_tag>{
	typedef matrix<T,L> type;
};

template<class T>
struct matrix_temporary_type<T,unknown_orientation,dense_random_access_iterator_tag, cpu_tag>{
	typedef matrix<T,row_major> type;
};

} //namespace aBLAS
#endif
