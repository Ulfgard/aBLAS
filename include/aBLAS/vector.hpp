/*!
 * 
 *
 * \brief       Implements the dense vector container class.
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
#ifndef ABLAS_VECTOR_HPP
#define ABLAS_VECTOR_HPP

#include "assignment.hpp"
#include "detail/iterator.hpp"

#include <memory>
#include <boost/container/vector.hpp>

namespace aBLAS {
namespace detail{
	
template<class T>	
struct dense_vector_state{
	typedef boost::container::vector<T> storage_type;//better handling of std::vector<bool>...
	typedef storage_type const const_storage_type;
	typedef typename storage_type::size_type size_type;
	typedef typename storage_type::value_type value_type;
	storage_type data;
	mutable scheduling::dependency_node dependencies;
	
	dense_vector_state(){}
	dense_vector_state(size_type size):data(size){}
	dense_vector_state(size_type size, value_type init):data(size,init){}
	template<class Iter>
	dense_vector_state(Iter begin, Iter end):data(begin,end){}
};
	
template<class SharedState>
class dense_vector_base{
protected:
	void set_state(SharedState* state){
		m_internals = state;
	}
	
	template<class> friend class dense_vector_base;
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
	typedef typename storage_type::reference reference;

	typedef std::size_t index_type;
	typedef dense_tag storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction
	dense_vector_base(){}
		
	//allows conversion non-const->const
	template<class S> 
	dense_vector_base(dense_vector_base<S> const& state):m_internals(state.m_internals){}


	/// \brief Return the size of the vector.
	size_type size() const {
		return storage().size();
	}
	
	/// \brief Return true if the vector is empty
	/// \return \c true if empty, \c false otherwise
	bool empty() const {
		return storage().empty();
	}
	
	/// \brief Clear the vector, i.e. set all values to the \c zero value.
	void clear() {
		//running via the scheduler is a lot of overhead for a simple operation
		//so if no kernels are in flight, we can just call std::fill directly
		//otherwise, we have to enqueue it as a kernel
		if(is_ready()){
			std::fill(storage().begin(), storage.end(), value_type/*zero*/());
		}else{
			dense_vector_base closure(*this);
			system::scheduler().spawn([closure](){
				std::fill(closure.begin(), closure.end(), value_type/*zero*/());
			},dependencies());
		}
	}

	// ---------
	// Dense low level interface
	// ---------
	///\brief Returns the internal vector storage
	///
	/// Grants low-level access to the vectors dense_vector_state. Elements storage()[0]...storage()[size()-1] are valid.
	storage_type& storage(){
		return m_internals->data;
	}
	
	///\brief Returns the internal vector storage
	///
	/// Grants low-level access to the vectors dense_vector_state. Elements storage()[0]...storage()[size()-1] are valid.
	const_storage_type& storage()const{
		return m_internals->data;
	}
	
	///\brief Returns the stride between the elements in storage()
	///
	/// In general elements of dense storage entities are spaced like storage()[offset()+i*stride()] for i=0,...,size()-1
	/// However for vector stride is guaranteed to be 1 and offset is 0.
	difference_type stride()const{
		return 1;
	}
	
	///\brief Returns the offset from the start of storage()
	///
	/// In general elements of dense storage entities are spaced like storage()[offset()+i*stride()] for i=0,...,size()-1
	/// However for vector stride is guaranteed to be 1 and offset is 0.
	difference_type offset()const{
		return 0;
	}
	
	// ---------
	// Async Interface
	// ---------
	
	/// \brief Returns true if this vector does not wait for operations to complete
	bool is_ready(){
		return !m_internals || m_internals->dependencies.is_ready();
	}
	
	/// \brief Blocks this thread until all kernels are computed.
	///
	/// Be aware that when other threads enqueue kernels in parallel while wait()
	/// is called, wait() can not guarantee that is_ready() is true after wait() returns.
	/// Thus, writing access to the vector after wait() has been called is undefined when there
	/// are other threads using the vector.
	void wait(){
		m_internals->dependencies.wait();
	}
	
	///\brief Returns the dependices of this vector.
	scheduling::dependency_node& dependencies() const{
		return m_internals->dependencies;
	}
	

	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// Return a const reference to the element \f$i\f$. With some compilers, this notation will be faster than \c operator[]
	/// \param i index of the element
	const_reference operator()(index_type i) const {
		return storage()[i];
	}

	/// \brief Return a reference to the element \f$i\f$
	/// Return a reference to the element \f$i\f$. With some compilers, this notation will be faster than \c operator[]
	/// \param i index of the element
	reference operator()(index_type i) {
		return storage()[i];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator [](index_type i) const {
		return (*this)(i);
	}

	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator [](index_type i) {
		return (*this)(i);
	}
	
	///\brief Returns the first element of the vector
	reference front(){
		return storage()[0];
	}
	///\brief Returns the first element of the vector
	const_reference front()const{
		return storage()[0];
	}
	///\brief Returns the last element of the vector
	reference back(){
		return storage()[size()-1];
	}
	///\brief Returns the last element of the vector
	const_reference back()const{
		return storage()[size()-1];
	}

	// Iterator types
private:
	typedef typename boost::mpl::if_<boost::is_const<storage_type>,
	        typename storage_type::const_iterator,
	        typename storage_type::iterator>::type storage_iterator;
public:
	typedef dense_storage_iterator<storage_iterator> iterator;
	typedef dense_storage_iterator<typename storage_type::const_iterator> const_iterator;
	
	/// \brief return an iterator on the first element of the vector
	const_iterator cbegin() const {
		return const_iterator(storage().begin(),0);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator cend() const {
		return const_iterator(storage().begin(),size());
	}

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return cbegin();
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return cend();
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return iterator(storage().begin(),0);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(storage().begin(),size());
	}
private:
	SharedState* m_internals;
};
} //namespace detail

/// \brief A dense vector of values of type \c T.
///
/// For a \f$n\f$-dimensional vector \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
/// to the \f$i\f$-th element of the container. 
///
/// \tparam T type of the objects stored in the vector (like int, double, complex,...)
template<class T>
class vector:public vector_expression<vector<T>, cpu_tag >, public detail::dense_vector_base<detail::dense_vector_state<T> > {
private:
	typedef typename detail::dense_vector_base<detail::dense_vector_state<T> > base;

	struct closure_type_base
	: public vector_expression<closure_type_base, cpu_tag >,
	  public base
	{
		typedef typename vector<T>::const_closure_type const_closure_type;
		typedef typename vector<T>::closure_type closure_type;
		closure_type_base(vector const& v):base(v){}
			
		using base::operator();
			
		// -------------------
		// Assignment operators
		// -------------------
		
		/// \brief Operator=
		closure_type_base& operator = (closure_type_base const& v) {
			ABLAS_SIZE_CHECK(v.size() == this->size());
			assign(*this,v);
			return *this;
		}

		/// \brief Assign the result of a vector_expression to the vector
		/// \param v is a const reference to the vector_expression
		/// \return a reference to the resulting vector
		template<class V>
		closure_type_base& operator = (vector_expression<V, cpu_tag> const& v) {
			ABLAS_SIZE_CHECK(v.size() ==  this->size());
			assign(*this,v);
			return *this;
		}
		
		
	};
	
	struct const_closure_type_base
	: public vector_expression<const_closure_type_base, cpu_tag >,
	  public detail::dense_vector_base<detail::dense_vector_state<T> const >
	{
		typedef typename vector<T>::const_closure_type const_closure_type;
		typedef typename vector<T>::const_closure_type closure_type;
		
		using detail::dense_vector_base<detail::dense_vector_state<T> const >::operator();

		const_closure_type_base(vector const& v):detail::dense_vector_base<detail::dense_vector_state<T> const >(v){}
		//constructor for non-const->const copying
		const_closure_type_base(closure_type_base const& c):detail::dense_vector_base<detail::dense_vector_state<T> const >(c){}
	};
public:
	typedef typename base::size_type size_type;
	typedef typename base::value_type value_type;
	typedef closure_type_base closure_type;
	typedef const_closure_type_base const_closure_type;
	using base::is_ready;
	using base::storage;
	using base::set_state;
	using base::size;
	using base::operator();
	

	// Construction and destruction

	/// \brief Constructor of a vector
	/// By default it is empty, i.e. \c size()==0.
	vector():m_internals(new detail::dense_vector_state<T>()){
		set_state(m_internals.get());
	}

	/// \brief Constructor of a vector with a predefined size
	/// \param size initial size of the vector
	explicit vector(size_type size):m_internals(new detail::dense_vector_state<T>(size)){
		set_state(m_internals.get());
	}
		
	/// \brief Constructs the vector from a predefined range
	template<class Iter>
	vector(Iter begin, Iter end):m_internals(new detail::dense_vector_state<T>(begin,end)){
		set_state(m_internals.get());
	}

	/// \brief Constructor of a vector with a predefined size with all elements initialized to an initial value
	/// \param size of the vector
	/// \param init value to assign to each element of the vector
	vector(size_type size, value_type init):m_internals(new detail::dense_vector_state<T>(size,init)){
		set_state(m_internals.get());
	}

	/// \brief Copy-constructor of a vector
	/// \param v is the vector to be duplicated
	vector(vector const& v):m_internals(new detail::dense_vector_state<T>(v.size())) {
		set_state(m_internals.get());
		if(v.is_ready())//v has no kernels in flight, just copy
			storage() = v.storage();
		else
			assign(*this,v);//start assignment kernel
	}
		
	/// \brief Move Constructor
	///
	///Moving a vector with active kernels is a well defined operation and guaranteed to work and non-blocking.
	vector(vector && v): m_internals(std::move(v.m_internals)){
		set_state(m_internals.get());
	}

	/// \brief Creates a vector from a vector_expression
	/// \param v the vector_expression which values will be assigned to the vector
	template<class V>
	vector(vector_expression<V, cpu_tag> const& v)
		:m_internals(new detail::dense_vector_state<T>(v().size())) {
		set_state(m_internals.get());
		assign(*this, v);
	}
	
	~vector(){
		//if this still owns memory and there are still kernels in flight, transfer ownership to the scheduler
		//this delays destruction until the last kernel using *this is finished
		if(!is_ready())
			system::scheduler().make_closure_variable(*this);
	}
	
	// -------------------
	// Assignment operators
	// -------------------
	
	/// \brief Operator=
	vector& operator = (vector const& v) {
		//if this vector is not used, we do not need to create a copy
		if(is_ready()){
			storage().resize(v.size());
			if(v.is_ready())//v has no kernels in flight, just copy
				storage() = v.storage();
			else
				assign(*this,v);//start assignment kernel
		}else{
			vector temporary(v);//start assignment kernel in temporary
			swap(*this,temporary);//swap dense_vector_state, destructor of the temporary will transfer ownership to the scheduler
		}
		return *this;
	}
	
	/// \brief Move Operator=
	vector& operator = (vector && v) {
		//if this vector is in use, we have to transfer ownership to the scheduler
		if(!is_ready())
			system::scheduler().make_closure_variable(std::move(*this));
		m_internals = std::move(v.m_internals);
		set_state(m_internals.get());
		
		return *this;
	}

	/// \brief Assign the result of a vector_expression to the vector
	///
	/// This operator assumes that the expressions are aliasing and thus always stores 
	/// the result of the expression in a temporary before copying
	/// \param v is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class V>
	vector& operator = (vector_expression<V, cpu_tag> const& v) {
		vector temporary(v);
		swap(*this,temporary);
		return *this;
	}
	
	/// \brief Resizes the vector
	///
	///The value of the elements after resize are undefined.
	///Be aware that resizing while kernels are in flight leads
	///to the creation of a new variable.
	void resize(size_type new_size){
		if(new_size == size())
			return;
		if(is_ready()){
			storage().resize(new_size);
		}else{
			//there are still kernels using this vector, so create a new variable and let the scheduler handle everything
			vector temporary(new_size);
			swap(*this,temporary);
		}
	}
	
	/// \brief Swap the content of two vectors
	/// \param v1 is the first vector. It takes values from v2
	/// \param v2 is the second vector It takes values from v1
	friend void swap(vector& v1, vector& v2) {
		v1.m_internals.swap(v2.m_internals);
		std::swap(static_cast<base&>(v1),static_cast<base&>(v2));
	}
private:
	std::unique_ptr<detail::dense_vector_state<T> > m_internals;
};

template<class T>
struct vector_temporary_type<T,dense_random_access_iterator_tag, cpu_tag>{
	typedef vector<T> type;
};

} //namespace aBLAS

#endif
