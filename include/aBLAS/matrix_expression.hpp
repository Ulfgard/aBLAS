#ifndef ABLAS_MATRIX_EXPRESSION_HPP
#define ABLAS_MATRIX_EXPRESSION_HPP

#include <boost/type_traits/is_convertible.hpp> 
#include <boost/utility/enable_if.hpp>

#include "assignment.hpp"
#include "matrix_proxy.hpp"
#include "detail/iterator.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemv.hpp"

namespace aBLAS {
	
/// \brief Matrix expression being asynchronously evaluated
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
typename matrix_temporary<E>::type async(matrix_expression<E,Device> const& e){
	return typename matrix_temporary<E>::type(e);
}

/** \brief A matrix with all values of type \c T equal to the same value
 */
template<class T, class Device>
class scalar_matrix:public matrix_expression<scalar_matrix<T, Device>, Device > {
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef const T& const_reference;
	typedef const_reference reference;

	typedef std::size_t index_type;
	typedef scalar_matrix const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename boost::mpl::if_<
		boost::is_same<Device,cpu_tag>,
		elementwise_tag,
		blockwise_tag
	>::type evaluation_category;
	// Construction and destruction
	scalar_matrix():
		m_size1(0), m_size2(0), m_value() {}
	scalar_matrix(size_type size1, size_type size2, const value_type& value = value_type(1)):
		m_size1(size1), m_size2(size2), m_value(value) {}
	scalar_matrix(const scalar_matrix& m):
		m_size1(m.m_size1), m_size2(m.m_size2), m_value(m.m_value) {}
		
	// Accessors
	size_type size1() const {
		return m_size1;
	}
	size_type size2() const {
		return m_size2;
	}
	
	std::vector<scheduling::dependency_node*> dependencies()const{
		return std::vector<scheduling::dependency_node*>();
	}

	// Element access
	const_reference operator()(size_type /*i*/, size_type /*j*/) const {
		return m_value;
	}
			
	//computation kernels
	template<class MatA>
	void assign_to(matrix_expression<MatA,cpu_tag>& A, value_type alpha = value_type(1) )const{
		typename MatA::closure_type A_closure(A());
		value_type t = alpha * m_value;
		system::scheduler().spawn([t, A_closure](){
			kernels::assign<scalar_plus_assign>(A_closure,t);
		},A().dependencies());
	}
	template<class MatA>
	void plus_assign_to(matrix_expression<MatA,cpu_tag>& A, value_type alpha = value_type(1) )const{
		A() += alpha * m_value;
	}


	//Iterators
	typedef constant_iterator<value_type> const_row_iterator;
	typedef constant_iterator<value_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(std::size_t i) const {
		return const_row_iterator(0, m_value);
	}
	const_row_iterator row_end(std::size_t i) const {
		return const_row_iterator(size2(), m_value);
	}
	
	const_row_iterator column_begin(std::size_t j) const {
		return const_row_iterator(0, m_value);
	}
	const_row_iterator column_end(std::size_t j) const {
		return const_row_iterator(size1(), m_value);
	}
private:
	size_type m_size1;
	size_type m_size2;
	value_type m_value;
};

///brief repeats a single element to form a matrix  of size rows x columns
///
///@param scalar the value which is repeated
///@param rows the number of rows of the resulting vector
///@param columns the number of columns of the resulting vector
template<class T, class Device>
typename boost::enable_if<boost::is_arithmetic<T>, scalar_matrix<T, Device> >::type
repeat(T scalar, std::size_t rows, std::size_t columns){
	return scalar_matrix<T, Device>(rows, columns, scalar);
}

template<class E>
class matrix_scalar_multiply:public matrix_expression<matrix_scalar_multiply<E>, typename E::device_category > {
private:
	typedef typename E::const_row_iterator const_subrow_iterator_type;
	typedef typename E::const_column_iterator const_subcolumn_iterator_type;
	typedef scalar_multiply1<typename E::value_type, typename E::value_type> functor_type;
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename functor_type::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;

	typedef typename E::index_type index_type;

	typedef matrix_scalar_multiply const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E::orientation orientation;
	typedef unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_category device_category;
private:
	expression_closure_type m_expression;
	value_type m_scalar;
public:

	// Construction and destruction
	matrix_scalar_multiply(matrix_expression<E,device_category> const &e, value_type scalar):
		m_expression(e()), m_scalar(scalar){}

	// Accessors
	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}
	
	auto dependencies()const -> decltype(this->m_expression.dependencies()){
		return m_expression.dependencies();
	}

	// Element access
	const_reference operator()(index_type i, index_type j) const {
		return m_scalar * m_expression(i, j);
	}
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX,device_category>& X, value_type alpha = value_type(1) )const{
		m_expression.assign_to(X,alpha*m_scalar);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX,device_category>& X, value_type alpha = value_type(1) )const{
		m_expression.plus_assign_to(X,alpha*m_scalar);
	}

	// Iterator types
	typedef transform_iterator<typename E::const_row_iterator, functor_type> const_row_iterator;
	typedef transform_iterator<typename E::const_column_iterator, functor_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(m_expression.row_begin(i),functor_type(m_scalar));
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_expression.row_end(i),functor_type(m_scalar));
	}

	const_column_iterator column_begin(index_type i) const {
		return const_row_iterator(m_expression.column_begin(i),functor_type(m_scalar));
	}
	const_column_iterator column_end(index_type i) const {
		return const_row_iterator(m_expression.column_end(i),functor_type(m_scalar));
	}
};

template<class E, class T, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type >,
        matrix_scalar_multiply<E> 
>::type
operator* (matrix_expression<E, Device> const& e, T scalar){
	return matrix_scalar_multiply<E>(e(), typename E::value_type(scalar));
}

template<class T, class E, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type >,
        matrix_scalar_multiply<E> 
>::type
operator* (T scalar, matrix_expression<E, Device> const& e){
	return matrix_scalar_multiply<E>(e(), typename E::value_type(scalar));
}

template<class E, class Device>
matrix_scalar_multiply<E> operator-(matrix_expression<E, Device> const& e){
	return matrix_scalar_multiply<E>(e(), typename E::value_type(-1));
}

template<class E1, class E2>
class matrix_addition: public matrix_expression<matrix_addition<E1, E2>, typename E1::device_category > {
private:
	typedef scalar_binary_plus<
		typename E1::value_type,
		typename E2::value_type
	> functor_type;
public:
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

	typedef typename E1::size_type size_type;
	typedef typename E1::difference_type difference_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef typename E1::index_type index_type;

	typedef const matrix_addition<E1, E2> const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E1::orientation orientation;
	typedef unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;
	typedef  typename E1::device_category device_category;

        // Construction
        matrix_addition(
		lhs_closure_type const& e1,
		rhs_closure_type const& e2
	): m_lhs (e1), m_rhs (e2){}

        // Accessors
        size_type size1 () const {
		return m_lhs.size1();
        }
        size_type size2 () const {
		return m_lhs.size2();
        }
	
	std::vector<scheduling::dependency_node*> dependencies()const{
		return gather_dependencies(m_lhs.dependencies(),m_rhs.dependencies());
	}

        const_reference operator () (index_type i, index_type j) const {
		return m_lhs(i, j) + m_rhs(i,j);
        }
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX,device_category>& X, value_type alpha = value_type(1) )const{
		assign(X,m_lhs, alpha);
		plus_assign(X,m_rhs, alpha);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX,device_category>& X, value_type alpha = value_type(1) )const{
		plus_assign(X,alpha * m_lhs, alpha);
		plus_assign(X,alpha * m_rhs, alpha);
	}

	// Iterator types
private:
	typedef typename E1::const_row_iterator const_row_iterator1_type;
	typedef typename E1::const_column_iterator const_row_column_iterator_type;
	typedef typename E2::const_row_iterator const_column_iterator1_type;
	typedef typename E2::const_column_iterator const_column_iterator2_type;
public:
	typedef binary_transform_iterator<
		typename E1::const_row_iterator,
		typename E2::const_row_iterator,
		functor_type
	> const_row_iterator;
	typedef binary_transform_iterator<
		typename E1::const_column_iterator,
		typename E2::const_column_iterator,
		functor_type
	> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(std::size_t i) const {
		return const_row_iterator (functor_type(),
			m_lhs.row_begin(i),m_lhs.row_end(i),
			m_rhs.row_begin(i),m_rhs.row_end(i)
		);
	}
	const_row_iterator row_end(std::size_t i) const {
		return const_row_iterator (functor_type(),
			m_lhs.row_end(i),m_lhs.row_end(i),
			m_rhs.row_end(i),m_rhs.row_end(i)
		);
	}

	const_column_iterator column_begin(std::size_t j) const {
		return const_column_iterator (functor_type(),
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}
	const_column_iterator column_end(std::size_t j) const {
		return const_column_iterator (functor_type(),
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}

private:
	lhs_closure_type m_lhs;
        rhs_closure_type m_rhs;
	functor_type m_functor;
};

///\brief Adds two Matrices
template<class E1, class E2, class Device>
matrix_addition<E1, E2> operator+ (
	matrix_expression<E1, Device> const& e1,
	matrix_expression<E2, Device> const& e2
){
	return matrix_addition<E1, E2>(e1(),e2());
}

///\brief Subtracts two Matrices
template<class E1, class E2, class Device>
matrix_addition<E1, matrix_scalar_multiply<E2> >
operator- (
	matrix_expression<E1,Device> const& e1,
	matrix_expression<E2,Device> const& e2
){
	return matrix_addition<E1, matrix_scalar_multiply<E2> >(e1(),-e2());
}

///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class E, class T, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type>, 
	matrix_addition<E, scalar_matrix<T, Device> >
>::type operator+ (
	matrix_expression<E, Device> const& e,
	T t
){
	return e + scalar_matrix<T, Device>(e().size1(),e().size2(),t);
}

///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class T, class E, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type>,
	matrix_addition<E, scalar_matrix<T, Device> >
>::type operator+ (
	T t,
	matrix_expression<E, Device> const& e
){
	return e + scalar_matrix<T,Device>(e().size1(),e().size2(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant matrix from a matrix.
template<class E, class T, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type> ,
	matrix_addition<E, matrix_scalar_multiply<scalar_matrix<T, Device> > >
>::type operator- (
	matrix_expression<E, Device> const& e,
	T t
){
	return e - scalar_matrix<T, Device>(e().size1(),e().size2(),t);
}

///\brief Subtracts a matrix from a scalar which is interpreted as a constant matrix
template<class E, class T, class Device>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type>,
	matrix_addition<scalar_matrix<T, Device>, matrix_scalar_multiply<E> >
>::type operator- (
	T t,
	matrix_expression<E, Device> const& e
){
	return scalar_matrix<T, Device>(e().size1(),e().size2(),t) - e;
}

template<class MatA, class VecV>
class matrix_vector_prod:
	public vector_expression<matrix_vector_prod<MatA, VecV>, typename MatA::device_category > {
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
	typedef typename VecV::const_closure_type vector_closure_type;
public:
	typedef typename promote_traits<
		typename MatA::value_type,
		typename VecV::value_type
	>::promote_type value_type;
	typedef typename MatA::size_type size_type;
	typedef typename MatA::difference_type difference_type;
	typedef typename MatA::index_type index_type;

	typedef matrix_vector_prod<MatA, VecV> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef blockwise_tag evaluation_category;
	typedef  typename MatA::device_category device_category;


	//FIXME: This workaround is required to be able to generate
	// temporary vectors
	typedef typename MatA::const_row_iterator const_iterator;
	typedef const_iterator iterator;

	// Construction and destruction
	matrix_vector_prod(
		matrix_closure_type const& matrix,
		vector_closure_type  const& vector
	):m_matrix(matrix), m_vector(vector) {}

	size_type size() const {
		return m_matrix.size1();
	}
	
	matrix_closure_type const& matrix() const {
		return m_matrix;
	}
	vector_closure_type const& vector() const {
		return m_vector;
	}
	
	std::vector<scheduling::dependency_node*> dependencies()const{
		return std::vector<scheduling::dependency_node*>();
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_category>& x, value_type alpha = value_type(1) )const{
		x().clear();
		plus_assign_to(x,alpha);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_category>& x, value_type alpha = value_type(1) )const{
		//dispatch based on whether the arguments require the creation of intermediate results
		plus_assign_to(x(),alpha, typename matrix_closure_type::evaluation_category(), typename vector_closure_type::evaluation_category());
	}
	
private:
	template<class VecX>
	void plus_assign_to(VecX& x, value_type alpha, elementwise_tag, elementwise_tag)const{
		start_kernel(x,alpha,m_matrix, m_vector, device_category());
	}
	
	template<class VecX>
	void plus_assign_to(VecX& x, value_type alpha, blockwise_tag, elementwise_tag)const{
		typedef typename matrix_temporary<matrix_closure_type>::type Temporary;
		system::scheduler().create_closure(
			Temporary(m_matrix.size1(),m_matrix.size2()),
			[this, &x,alpha](Temporary& temporary){
				assign(temporary,m_matrix);
				start_kernel(x,alpha,temporary, m_vector, device_category());
			}
		);
	}
	
	template<class VecX>
	void plus_assign_to(VecX& x, value_type alpha, elementwise_tag, blockwise_tag)const{
		typedef typename vector_temporary<vector_closure_type>::type Temporary;
		system::scheduler().create_closure(
			Temporary(m_vector.size()),
			[this, &x,alpha](Temporary& temporary){
				assign(temporary,m_matrix);
				start_kernel(x,alpha,m_matrix, temporary, device_category());
			}
		);
	}
	
	template<class VecX>
	void plus_assign_to(VecX& x, value_type alpha, blockwise_tag, blockwise_tag)const{
		typedef typename matrix_temporary<matrix_closure_type>::type TemporaryM;
		typedef typename vector_temporary<vector_closure_type>::type TemporaryV;
		system::scheduler().create_closure(
			TemporaryM(m_matrix.size1(),m_matrix.size2()),
			TemporaryV(m_vector.size()),
			[this, &x,alpha](TemporaryM& tempM, TemporaryV& tempV){
				assign(tempM,m_matrix);
				assign(tempV,m_vector);
				start_kernel(x,alpha,tempM, tempV, device_category());
			}
		);
	}

	//the actual kernel calling routine (cpu version)
	template<class VecX, class MatrixA, class ArgV>
	void start_kernel(VecX& x, value_type alpha, MatrixA const& A, ArgV const& v,cpu_tag)const{
		typename VecX::closure_type x_closure(x());
		typename ArgV::const_closure_type v_closure(v());
		typename MatrixA::const_closure_type A_closure(A());
		system::scheduler().spawn([alpha, x_closure, v_closure, A_closure](){
			kernels::gemv(A_closure, v_closure, x_closure, alpha);
		},x.dependencies(),v.dependencies(),A.dependencies());
	}

	matrix_closure_type m_matrix;
	vector_closure_type m_vector;
};


/// \brief computes the matrix-vector product x+=Av
template<class MatA, class VecV,class Device>
matrix_vector_prod<MatA,VecV> prod(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecV, Device> const& v
) {
	return matrix_vector_prod<MatA,VecV>(A(),v());
}

/// \brief computes the matrix-vector product x+=v^TA
template<class MatA, class VecV, class Device>
matrix_vector_prod<matrix_transpose<MatA>,VecV>
prod(vector_expression<VecV, Device> const& v,matrix_expression<MatA, Device> const& A) {
	//compute it using the identity (v^TA)^T= A^Tv
	return matrix_vector_prod<matrix_transpose<MatA>,VecV>(trans(A),v());
}

//matrix-matrix prod
template<class MatA, class MatB>
class matrix_matrix_prod: public matrix_expression<matrix_matrix_prod<MatA, MatB>, typename MatA::device_category > {
public:
	typedef typename MatA::const_closure_type matrix_closure_typeA;
	typedef typename MatB::const_closure_type matrix_closure_typeB;
public:
	typedef typename promote_traits<
		typename MatA::value_type,
		typename MatB::value_type
	>::promote_type value_type;
	typedef typename MatA::size_type size_type;
	typedef typename MatA::difference_type difference_type;
	typedef typename MatA::index_type index_type;

	typedef matrix_matrix_prod<MatA, MatB> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef blockwise_tag evaluation_category;
	typedef unknown_orientation orientation;
	typedef typename MatA::device_category device_category;


	//FIXME: This workaround is required to be able to generate
	// temporary matrices
	typedef typename MatA::const_row_iterator const_row_iterator;
	typedef typename MatA::const_column_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	// Construction and destruction
	matrix_matrix_prod(
		matrix_closure_typeA const& matrixA,
		matrix_closure_typeB const& matrixB
	):m_matrixA(matrixA), m_matrixB(matrixB) {}

	size_type size1() const {
		return m_matrixA.size1();
	}
	size_type size2() const {
		return m_matrixB.size2();
	}
	
	matrix_closure_typeA const& matrixA() const {
		return m_matrixA;
	}
	matrix_closure_typeB const& matrixB() const {
		return m_matrixB;
	}
	
	std::vector<scheduling::dependency_node*> dependencies()const{
		return std::vector<scheduling::dependency_node*>();
	}
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX,device_category>& X, value_type alpha = value_type(1) )const{
		X().clear();
		plus_assign_to(X,alpha);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_category>& X, value_type alpha = value_type(1) )const{
		plus_assign_to(X(),alpha, typename matrix_closure_typeA::evaluation_category(), typename matrix_closure_typeB::evaluation_category());
	}
	
private:
	template<class MatX>
	void plus_assign_to(MatX& X, value_type alpha, elementwise_tag, elementwise_tag)const{
		start_kernel(X,alpha,m_matrixA, m_matrixB, device_category());
	}
	
	template<class MatX>
	void plus_assign_to(MatX& X, value_type alpha, blockwise_tag, elementwise_tag)const{
		typedef typename matrix_temporary<matrix_closure_typeA>::type Temporary;
		system::scheduler().create_closure(
			Temporary(m_matrixA.size1(),m_matrixA.size2()),
			[this, &X,alpha](Temporary& temporary){
				assign(temporary,m_matrixA);
				start_kernel(X,alpha,temporary, m_matrixB, device_category());
			}
		);
	}
	
	template<class MatX>
	void plus_assign_to(MatX& X, value_type alpha, elementwise_tag, blockwise_tag)const{
		typedef typename matrix_temporary<matrix_closure_typeB>::type Temporary;
		system::scheduler().create_closure(
			Temporary(m_matrixA.size1(),m_matrixA.size2()),
			[this, &X,alpha](Temporary& temporary){
				assign(temporary,m_matrixB);
				start_kernel(X,alpha,m_matrixA, temporary, device_category());
			}
		);
	}
	
	template<class MatX>
	void plus_assign_to(MatX& X, value_type alpha, blockwise_tag, blockwise_tag)const{
		typedef typename matrix_temporary<matrix_closure_typeA>::type TemporaryA;
		typedef typename matrix_temporary<matrix_closure_typeB>::type TemporaryB;
		system::scheduler().create_closure(
			TemporaryA(m_matrixA.size1(),m_matrixA.size2()),
			TemporaryB(m_matrixB.size1(),m_matrixB.size2()),
			[this, &X,alpha](TemporaryA& tempA, TemporaryB& tempB){
				assign(tempA, m_matrixA);
				assign(tempB, m_matrixB);
				start_kernel(X,alpha,tempA, tempB, device_category());
			}
		);
	}

	//the actual kernel calling routine (cpu version)
	template<class MatrixX, class MatrixA, class MatrixB>
	void start_kernel(MatrixX& X, value_type alpha, MatrixA const& A, MatrixB const& B,cpu_tag)const{
		typename MatrixX::closure_type X_closure(X);
		typename MatrixA::const_closure_type A_closure(A);
		typename MatrixB::const_closure_type B_closure(B);
		system::scheduler().spawn([alpha, X_closure, A_closure, B_closure]()mutable{
			kernels::gemm(A_closure, B_closure, X_closure, alpha);
		},X.dependencies(),A.dependencies(),B.dependencies());
	}
	
	matrix_closure_typeA m_matrixA;
	matrix_closure_typeB m_matrixB;
};

/// \brief computes the matrix-matrix product X+=AB
template<class MatA, class MatB, class Device>
matrix_matrix_prod<MatA,MatB> prod(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) {
	return matrix_matrix_prod<MatA,MatB>(A(),B());
}

}

#endif
