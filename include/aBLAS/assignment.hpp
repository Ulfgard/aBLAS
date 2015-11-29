/*!
 * 
 *
 * \brief       Assignment operators
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

#ifndef ABLAS_ASSIGNMENT_HPP
#define ABLAS_ASSIGNMENT_HPP

#include "kernels/matrix_assign.hpp"
#include "kernels/vector_assign.hpp"
#include "detail/traits.hpp"
#include "scheduling/scheduling.hpp"

namespace aBLAS{
	
///gather dependencies(needs own file to live in)
std::vector<scheduling::dependency_node*> gather_dependencies(
	std::vector<scheduling::dependency_node*> list1,
	std::vector<scheduling::dependency_node*>const& list2
){
	list1.insert(list1.end(),list2.begin(),list2.end());
	return std::move(list1);
}
std::vector<scheduling::dependency_node*> gather_dependencies(
	std::vector<scheduling::dependency_node*> list,
	scheduling::dependency_node& dep
){
	list.push_back(&dep);
	return list;
}
std::vector<scheduling::dependency_node*> gather_dependencies(
	scheduling::dependency_node& dep,
	std::vector<scheduling::dependency_node*> list
){
	list.push_back(&dep);
	return list;
}
std::vector<scheduling::dependency_node*> gather_dependencies(
	scheduling::dependency_node& dep1,
	scheduling::dependency_node& dep2
){
	return std::vector<scheduling::dependency_node*>({&dep1,&dep2});
}
/////////////////////////////////////////////////////////////////////////////////////
////// Vector Assign
////////////////////////////////////////////////////////////////////////////////////
	
namespace detail{
	template<class VecX, class VecV>
	void assign(
		vector_expression<VecX, cpu_tag>& x,
		vector_expression<VecV, cpu_tag> const& v,
		typename VecX::value_type alpha,
		elementwise_tag
	){
		typename VecX::closure_type x_closure(x());
		typename VecV::const_closure_type v_closure(v());
		system::scheduler().spawn([alpha, x_closure, v_closure]()mutable{
			kernels::assign<scalar_assign>(x_closure,v_closure,alpha);
		},x().dependencies(),v().dependencies());
	}
	template<class VecX, class VecV, class Device>
	void assign(
		vector_expression<VecX, Device>& x,
		vector_expression<VecV, Device> const& v,
		typename VecX::value_type alpha,
		blockwise_tag
	){
		v().assign_to(x,alpha);
	}
	template<class VecX, class VecV>
	void plus_assign(
		vector_expression<VecX, cpu_tag>& x,
		vector_expression<VecV, cpu_tag> const& v,
		typename VecX::value_type alpha,
		elementwise_tag
	){
		typename VecX::closure_type x_closure(x());
		typename VecV::const_closure_type v_closure(v());
		system::scheduler().spawn([alpha, x_closure, v_closure]()mutable{
			kernels::assign<scalar_plus_assign>(x_closure,v_closure,alpha);
		},x().dependencies(),v().dependencies());
	}
	template<class VecX, class VecV, class Device>
	void plus_assign(
		vector_expression<VecX, Device>& x,
		vector_expression<VecV, Device> const& v,
		typename VecX::value_type alpha,
		blockwise_tag
	){
		v().plus_assign_to(x,alpha);
	}
}
	

/// \brief Computes the assignment of the expressions x=v or x=alpha*v where alpha is a scalar. 
template<class VecX, class VecV, class Device>
void assign(
	vector_expression<VecX, Device>& x,
	vector_expression<VecV, Device> const& v,
	typename VecX::value_type alpha = typename VecX::value_type(1)
){
	ABLAS_SIZE_CHECK(x().size() == v().size());
	//dispatch to blockwise or elementwise evaluation
	detail::assign(x,v,alpha,typename VecV::evaluation_category());
}

/// \brief Computes the assignment of the expressions x+=v or x+=alpha*v where alpha is a scalar. 
template<class VecX, class VecV, class Device>
void plus_assign(
	vector_expression<VecX, Device>& x,
	vector_expression<VecV, Device> const& v,
	typename VecX::value_type alpha = typename VecX::value_type(1)
){
	ABLAS_SIZE_CHECK(x().size() == v().size());
	//dispatch to blockwise or elementwise evaluation
	detail::plus_assign(x,v,alpha,typename VecV::evaluation_category());
}
	
/////////////////////////////////////////////////////////////////////////////////////
////// Matrix Assign
////////////////////////////////////////////////////////////////////////////////////
	
namespace detail{
	template<class MatA, class MatB>
	void assign(
		matrix_expression<MatA, cpu_tag>& A,
		matrix_expression<MatB, cpu_tag> const& B,
		typename MatA::value_type const& alpha,
		elementwise_tag
	){
		typename MatA::closure_type A_closure(A());
		typename MatB::const_closure_type B_closure(B());
		system::scheduler().spawn([alpha, A_closure, B_closure]()mutable{
			kernels::assign<scalar_assign>(A_closure,B_closure,alpha);
		},A().dependencies(),B().dependencies());
	}
	template<class MatA, class MatB, class Device>
	void assign(
		matrix_expression<MatA, Device>& A,
		matrix_expression<MatB, Device> const& B,
		typename MatA::value_type const& alpha,
		blockwise_tag
	){
		B().assign_to(A,alpha);
	}
	template<class MatA, class MatB>
	void plus_assign(
		matrix_expression<MatA, cpu_tag>& A,
		matrix_expression<MatB, cpu_tag> const& B,
		typename MatA::value_type const& alpha,
		elementwise_tag
	){
		typename MatA::closure_type A_closure(A());
		typename MatB::const_closure_type B_closure(B());
		system::scheduler().spawn([alpha, A_closure, B_closure]()mutable{
			kernels::assign<scalar_plus_assign>(A_closure,B_closure,alpha);
		},A().dependencies(),B().dependencies());
	}
	template<class MatA, class MatB, class Device>
	void plus_assign(
		matrix_expression<MatA, Device>& A,
		matrix_expression<MatB, Device> const& B,
		typename MatA::value_type const& alpha,
		blockwise_tag
	){
		B().plus_assign_to(A,alpha);
	}
}
	

/// \brief Computes the assignment of the expressions A=B or A=alpha*B where alpha is a scalar. 
template<class MatA, class MatB, class Device>
void assign(
	matrix_expression<MatA, Device>& A,
	matrix_expression<MatB, Device> const& B,
	typename MatA::value_type alpha = typename MatA::value_type(1)
){
	ABLAS_SIZE_CHECK(A().size1() == B().size1());
	ABLAS_SIZE_CHECK(A().size2() == B().size2());
	detail::assign(A,B, alpha, typename MatB::evaluation_category());
}

/// \brief Computes the assignment of the expressions A+=B or A+=alpha*B where alpha is a scalar. 
template<class MatA, class MatB, class Device>
void plus_assign(
	matrix_expression<MatA, Device>& A,
	matrix_expression<MatB, Device> const& B,
	typename MatA::value_type alpha = typename MatA::value_type(1)
){
	ABLAS_SIZE_CHECK(A().size1() == B().size1());
	ABLAS_SIZE_CHECK(A().size2() == B().size2());
	detail::plus_assign(A,B, alpha, typename MatB::evaluation_category());
}

//////////////////////////////////////////////////////////////////////////////////////
///// Vector Operators
/////////////////////////////////////////////////////////////////////////////////////

/// \brief  Add-Assigns two vector expressions
///
/// Performs the operation x_i+=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)+=v to avoid this if A and B do not alias
template<class VecX, class VecV, class Device>
VecX& operator += (vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	ABLAS_SIZE_CHECK(x().size() == v().size());
	typedef typename vector_temporary<VecX>::type Temporary;
	system::scheduler().create_closure(
		Temporary(v.size()),
		[&x, &v](Temporary& temporary){
			assign(temporary,v);
			plus_assign(x,temporary);
		}
	);
	return x();
}

/// \brief  Subtract-Assigns two vector expressions
///
/// Performs the operation x_i-=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)-=v to avoid this if A and B do not alias
template<class VecX, class VecV, class Device>
VecX& operator-=(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	ABLAS_SIZE_CHECK(x().size() == v().size());
	typedef typename vector_temporary<VecX>::type Temporary;
	system::scheduler().create_closure(
		Temporary(v.size()),
		[&x, &v](Temporary& temporary){
			assign(temporary,v);
			plus_assign(x,temporary,typename VecX::value_type(-1));
		}
	);
	return x();
}

/// \brief  Adds a scalar to all elements of the vector
///
/// Performs the operation x_i += t for all elements.
template<class VecX>
VecX& operator+=(vector_expression<VecX, cpu_tag>& x, typename VecX::value_type t){
	typename VecX::closure_type x_closure(x());
	system::scheduler().spawn([t, x_closure]()mutable{
		kernels::assign<scalar_plus_assign>(x_closure,t);
	},x().dependencies());
	return x();
}

/// \brief  Subtracts a scalar from all elements of the vector
///
/// Performs the operation x_i += t for all elements.
template<class VecX>
VecX& operator-=(vector_expression<VecX, cpu_tag>& x, typename VecX::value_type t){
	typename VecX::closure_type x_closure(x());
	system::scheduler().spawn([t, x_closure]()mutable{
		kernels::assign<scalar_minus_assign>(x_closure,t);
	},x().dependencies());
	return x();
}

/// \brief  Multiplies a scalar with all elements of the vector
///
/// Performs the operation x_i *= t for all elements.
template<class VecX>
VecX& operator*=(vector_expression<VecX, cpu_tag>& x, typename VecX::value_type t){
	typename VecX::closure_type x_closure(x());
	system::scheduler().spawn([t, x_closure]()mutable{
		kernels::assign<scalar_multiply_assign>(x_closure,t);
	},x().dependencies());
	return x();
}

/// \brief  Divides all elements of the vector by a scalar
///
/// Performs the operation x_i /= t for all elements.
template<class VecX, class Device>
VecX& operator/=(vector_expression<VecX, Device>& x, typename VecX::value_type t){
	typename VecX::closure_type x_closure(x());
	system::scheduler().spawn([t, x_closure]()mutable{
		kernels::assign<scalar_divide_assign>(x_closure,t);
	},x().dependencies());
	return x();
}



//////////////////////////////////////////////////////////////////////////////////////
///// Matrix Operators
/////////////////////////////////////////////////////////////////////////////////////

/// \brief  Add-Assigns two matrix expressions
///
/// Performs the operation A_ij+=B_ij for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(A)+=B to avoid this if A and B do not alias
template<class MatA, class MatB, class Device>
MatA& operator+=(
	matrix_expression<MatA, Device>& A,
	matrix_expression<MatB, Device> const& B
){
	ABLAS_SIZE_CHECK(A().size1() == B().size1());
	ABLAS_SIZE_CHECK(A().size2() == B().size2());
	typedef typename matrix_temporary<MatA>::type Temporary;
	system::scheduler().create_closure(
		Temporary(A.size1(),A.size2()),
		[&A, &B](Temporary& temporary){
			assign(temporary,B);
			plus_assign(A,temporary);
		}
	);
	return A();
}

/// \brief  Subtract-Assigns two matrix expressions
///
/// Performs the operation A_ij-=B_ij for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(A)-=B to avoid this if A and B do not alias
template<class MatA, class MatB, class Device>
MatA& operator-=(
	matrix_expression<MatA, Device>& A,
	matrix_expression<MatB, Device> const& B
){
	ABLAS_SIZE_CHECK(A().size1() == B().size1());
	ABLAS_SIZE_CHECK(A().size2() == B().size2());
	typedef typename matrix_temporary<MatA>::type Temporary;
	system::scheduler().create_closure(
		Temporary(A.size1(),A.size2()),
		[&A, &B](Temporary& temporary){
			assign(temporary,B);
			plus_assign(A,temporary, typename MatA::value_type(-1));
		}
	);
	return A();
}

/// \brief  Adds a scalar to all elements of the matrix
///
/// Performs the operation A_ij += t for all elements.
template<class MatA>
MatA& operator+=(matrix_expression<MatA, cpu_tag>& A, typename MatA::value_type t){
	typename MatA::closure_type A_closure(A());
	system::scheduler().spawn([t, A_closure]()mutable{
		kernels::assign<scalar_plus_assign>(A_closure,t);
	},A().dependencies());
	return A();
}

/// \brief  Subtracts a scalar from all elements of the matrix
///
/// Performs the operation A_ij -= t for all elements.
template<class MatA>
MatA& operator-=(matrix_expression<MatA, cpu_tag>& A, typename MatA::value_type t){
	typename MatA::closure_type A_closure(A());
	system::scheduler().spawn([t, A_closure]()mutable{
		kernels::assign<scalar_minus_assign>(A_closure,t);
	},A().dependencies());
	return A();
}

/// \brief  Multiplies a scalar to all elements of the matrix
///
/// Performs the operation A_ij *= t for all elements.
template<class MatA>
MatA& operator*=(matrix_expression<MatA, cpu_tag>& A, typename MatA::value_type t){
	typename MatA::closure_type A_closure(A());
	system::scheduler().spawn([t, A_closure]()mutable{
		kernels::assign<scalar_multiply_assign>(A_closure,t);
	},A().dependencies());
	return A();
}

/// \brief  Divides all elements of the matrix by a scalar
///
/// Performs the operation A_ij /= t for all elements.
template<class MatA>
MatA& operator /=(matrix_expression<MatA, cpu_tag>& A, typename MatA::value_type t){
	typename MatA::closure_type A_closure(A());
	system::scheduler().spawn([t, A_closure]()mutable{
		kernels::assign<scalar_divide_assign>(A_closure,t);
	},A().dependencies());
	return A();
}
//////////////////////////////////////////////////////////////////////////////////////
///// Temporary Proxy Operators
/////////////////////////////////////////////////////////////////////////////////////

template<class T, class U>
temporary_proxy<T> operator+=(temporary_proxy<T> x, U const& arg){
	static_cast<T&>(x) += arg;
	return x;
}
template<class T, class U>
temporary_proxy<T> operator-=(temporary_proxy<T> x, U const& arg){
	static_cast<T&>(x) -= arg;
	return x;
}
template<class T, class U>
temporary_proxy<T> operator*=(temporary_proxy<T> x, U const& arg){
	static_cast<T&>(x) *= arg;
	return x;
}
template<class T, class U>
temporary_proxy<T> operator/=(temporary_proxy<T> x, U const& arg){
	static_cast<T&>(x) /= arg;
	return x;
}




//////////////////////////////////////////////////////////////////////////////////////
///// Noalias Assignment Proxies
/////////////////////////////////////////////////////////////////////////////////////
template<class C>
class noalias_proxy{
public:
	typedef typename C::closure_type closure_type;
	typedef typename C::value_type value_type;

	noalias_proxy(C &lval): m_lval(lval) {}

	noalias_proxy(const noalias_proxy &p):m_lval(p.m_lval) {}

	template <class E>
	closure_type &operator= (const E &e) {
		assign(m_lval, e);
		return m_lval;
	}

	template <class E>
	closure_type &operator+= (const E &e) {
		plus_assign(m_lval, e);
		return m_lval;
	}

	template <class E>
	closure_type &operator-= (const E &e) {
		minus_assign(m_lval, e);
		return m_lval;
	}
	
	template <class E>
	closure_type &operator*= (const E &e) {
		multiply_assign(m_lval, e);
		return m_lval;
	}

	template <class E>
	closure_type &operator/= (const E &e) {
		divide_assign(m_lval, e);
		return m_lval;
	}
	
	//this is not needed, but prevents errors when for example doing noalias(x)+=2;
	closure_type &operator+= (value_type t) {
		return m_lval += t;
	}

	//this is not needed, but prevents errors when for example doing noalias(x)-=2;
	closure_type &operator-= (value_type t) {
		return m_lval -= t;
	}
	
	//this is not needed, but prevents errors when for example doing noalias(x)*=2;
	closure_type &operator*= (value_type t) {
		return m_lval *= t;
	}

	//this is not needed, but prevents errors when for example doing noalias(x)/=2;
	closure_type &operator/= (value_type t) {
		return m_lval /= t;
	}

private:
	closure_type m_lval;
};

// Improve syntax of efficient assignment where no aliases of LHS appear on the RHS
//  noalias(lhs) = rhs_expression
template <class C, class Device>
noalias_proxy<C> noalias(matrix_expression<C, Device>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C, class Device>
noalias_proxy<C> noalias(vector_expression<C, Device>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C>
noalias_proxy<C> noalias(temporary_proxy<C> lvalue) {
	return noalias_proxy<C> (static_cast<C&>(lvalue));
}

}
#endif