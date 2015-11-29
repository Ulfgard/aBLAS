#define BOOST_TEST_MODULE aBLAS_assign
#include <boost/test/unit_test.hpp>


#include <aBLAS/kernels/vector_assign.hpp>
#include <aBLAS/kernels/matrix_assign.hpp>
#include <aBLAS/vector.hpp>
#include <aBLAS/matrix.hpp>


using namespace aBLAS;

template<class V1, class V2>
void checkVectorEqual(V1 const& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_EQUAL(v1(i),v2(i));
	}
}
template<class M1, class M2>
void checkMatrixEqual(M1 const& m1, M2 const& m2){
	BOOST_REQUIRE_EQUAL(m1.size1(),m2.size1());
	BOOST_REQUIRE_EQUAL(m1.size2(),m2.size2());
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			BOOST_CHECK_EQUAL(m1(i,j),m2(i,j));
		}
	}
}

BOOST_AUTO_TEST_SUITE (aBLAS_assign)

BOOST_AUTO_TEST_CASE( aBLAS_assign_vector_dense ){
	std::cout<<"testing dense vector assignment"<<std::endl;
	vector<unsigned int> source(10);
	
	for(std::size_t i = 0; i != 10; ++i){
		source(i) = 2*i+1;
	}
	
	//direct assignment kernel
	{
		std::cout<<"testing x=alpha*v"<<std::endl;
		vector<unsigned int> target(10,1);
		vector<unsigned int> result2(10);
		for(std::size_t i = 0; i != 10; ++i){
			result2(i) = 2*source(i);
		}
		kernels::assign<scalar_assign>(target,source,1);
		checkVectorEqual(target,source);
		kernels::assign<scalar_assign>(target,source,2);
		checkVectorEqual(target,result2);
	}
	//plus assignment kernel
	{
		std::cout<<"testing x+=alpha*v"<<std::endl;
		vector<unsigned int> target(10);
		vector<unsigned int> result(10);
		for(std::size_t i = 0; i != 10; ++i){
			target(i) = i+1;
			result(i) = target(i) + 2*source(i);
		}
		kernels::assign<scalar_plus_assign>(target,source,2);
		checkVectorEqual(target,result);
	}
	//scalar assignment
	{
		std::cout<<"testing x+=t"<<std::endl;
		vector<unsigned int> target(10);
		vector<unsigned int> result(10);
		for(std::size_t i = 0; i != 10; ++i){
			target(i) = i;
			result(i) = i+2;
		}
		kernels::assign<scalar_plus_assign>(target,2);
		checkVectorEqual(target,result);
	}
}

BOOST_AUTO_TEST_CASE( aBLAS_assign_matrix_dense ){
	std::cout<<"testing dense matrix assignment"<<std::endl;
	matrix<unsigned int,row_major> source_rm(10,20);
	matrix<unsigned int,column_major> source_cm(10,20);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 20; ++j){
			source_rm(i,j) = source_cm(i,j) = 2*i+j+1;
		}
	}
	
	//direct assignment kernel row-major
	{
		std::cout<<"testing A=alpha*B"<<std::endl;
		matrix<unsigned int,row_major> result2(10,20,0);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				result2(i,j) = 2*source_rm(i,j);
			}
		}
		
		//alpha=1
		{
		std::cout<<"rr"<<std::endl;
		matrix<unsigned int,row_major> target_rm(10,20,1);
		kernels::assign<scalar_assign>(target_rm,source_rm,1);
		checkMatrixEqual(target_rm,source_rm);
		}
		{
		std::cout<<"rc"<<std::endl;
		matrix<unsigned int,row_major> target_rm(10,20,1);
		kernels::assign<scalar_assign>(target_rm,source_cm,1);
		checkMatrixEqual(target_rm,source_rm);
		}
		{
		std::cout<<"cr"<<std::endl;
		matrix<unsigned int,column_major> target_cm(10,20,1);
		kernels::assign<scalar_assign>(target_cm,source_rm,1);
		checkMatrixEqual(target_cm,source_rm);
		}
		{
		std::cout<<"cc"<<std::endl;
		matrix<unsigned int,column_major> target_cm(10,20,1);
		kernels::assign<scalar_assign>(target_cm,source_cm,1);
		checkMatrixEqual(target_cm,source_rm);
		}
		//alpha=2
		{
		std::cout<<"rr"<<std::endl;
		matrix<unsigned int,row_major> target_rm(10,20,1);
		kernels::assign<scalar_assign>(target_rm,source_rm,2);
		checkMatrixEqual(target_rm,result2);
		}
		{
		std::cout<<"rc"<<std::endl;
		matrix<unsigned int,row_major> target_rm(10,20,1);
		kernels::assign<scalar_assign>(target_rm,source_cm,2);
		checkMatrixEqual(target_rm,result2);
		}
		{
		std::cout<<"cr"<<std::endl;
		matrix<unsigned int,column_major> target_cm(10,20,1);
		kernels::assign<scalar_assign>(target_cm,source_rm,2);
		checkMatrixEqual(target_cm,result2);
		}
		{
		std::cout<<"cc"<<std::endl;
		matrix<unsigned int,column_major> target_cm(10,20,1);
		kernels::assign<scalar_assign>(target_cm,source_cm,2);
		checkMatrixEqual(target_cm,result2);
		}
	}
	//plus assignment kernel
	{
		std::cout<<"testing A+=alpha*B"<<std::endl;
		matrix<unsigned int,row_major> result(10,20,0);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				result(i,j) = 2*source_rm(i,j) +i+j;
			}
		}
		
		{
		std::cout<<"rr"<<std::endl;
		matrix<unsigned int,row_major> target_rm(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target_rm(i,j) = i+j;
			}
		}
		kernels::assign<scalar_plus_assign>(target_rm,source_rm,2);
		checkMatrixEqual(target_rm,result);
		}
		{
		std::cout<<"rc"<<std::endl;
		matrix<unsigned int,row_major> target_rm(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target_rm(i,j) = i+j;
			}
		}
		kernels::assign<scalar_plus_assign>(target_rm,source_cm,2);
		checkMatrixEqual(target_rm,result);
		}
		{
		std::cout<<"cr"<<std::endl;
		matrix<unsigned int,row_major> target_cm(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target_cm(i,j) = i+j;
			}
		}
		kernels::assign<scalar_plus_assign>(target_cm,source_rm,2);
		checkMatrixEqual(target_cm,result);
		}
		{
		std::cout<<"cc"<<std::endl;
		matrix<unsigned int,row_major> target_cm(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target_cm(i,j) = i+j;
			}
		}
		kernels::assign<scalar_plus_assign>(target_cm,source_cm,2);
		checkMatrixEqual(target_cm,result);
		}
	}
	//scalar assignment
	{
		std::cout<<"testing A += t"<<std::endl;
		matrix<unsigned int,row_major> target_rm(10,20);
		matrix<unsigned int,row_major> target_cm(10,20);
		matrix<unsigned int,row_major> result(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target_rm(i,j) = target_cm(i,j) = i+j;
				result(i,j) = target_cm(i,j) + 2;
			}
		}
		kernels::assign<scalar_plus_assign>(target_rm,2);
		kernels::assign<scalar_plus_assign>(target_cm,2);
		checkMatrixEqual(target_rm,result);
		checkMatrixEqual(target_cm,result);
	}
}



BOOST_AUTO_TEST_SUITE_END()
