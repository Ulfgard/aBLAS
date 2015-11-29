#define BOOST_TEST_MODULE aBLAS_gemm
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <aBLAS/matrix.hpp>
#include <aBLAS/kernels/gemm.hpp>

using namespace aBLAS;

//we test using the textbook definition.
template<class Arg1, class Arg2, class Result>
void checkMatrixMatrixMultiply(Arg1 const& arg1, Arg2 const& arg2, Result const& result, double factor, double init = 0){
	BOOST_REQUIRE_EQUAL(arg1.size1(), result.size1());
	BOOST_REQUIRE_EQUAL(arg2.size2(), result.size2());
	
	for(std::size_t i = 0; i != arg1.size1(); ++i){
		for(std::size_t j = 0; j != arg2.size2(); ++j){
			double test_result = init;
			for(std::size_t k = 0; k != arg1.size2(); ++k){
				 test_result += factor * arg1(i,k)*arg2(k,j);
			}
			BOOST_CHECK_CLOSE(result(i,j), test_result,1.e-10);
		}
	}
}

BOOST_AUTO_TEST_SUITE (aBLAS_gemm)

BOOST_AUTO_TEST_CASE( aBLAS_gemm_dense_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	matrix<double,row_major> arg1rm(rows,middle);
	matrix<double,column_major> arg1cm(rows,middle);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != middle; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*middle+0.2*j;
		}
	}
	matrix<double,row_major> arg2rm(middle,columns);
	matrix<double,column_major> arg2cm(middle,columns);
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg2rm(i,j) = arg2cm(i,j) = i*columns+1.5*j;
		}
	}
	double alpha=-2.0;
	std::cout<<"\nchecking dense-dense gemm"<<std::endl;
	//test first expressions of the form A+=alpha*B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		kernels::gemm(arg1rm,arg2rm,resultrm,alpha);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,alpha,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		kernels::gemm(arg1rm,arg2rm,resultcm,alpha);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,alpha,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		kernels::gemm(arg1rm,arg2cm,resultrm,alpha);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,alpha,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		kernels::gemm(arg1rm,arg2cm,resultcm,alpha);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,alpha,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		kernels::gemm(arg1cm,arg2rm,resultrm,alpha);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,alpha,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		kernels::gemm(arg1cm,arg2rm,resultcm,alpha);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,alpha,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		kernels::gemm(arg1cm,arg2cm,resultrm,alpha);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,alpha,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		kernels::gemm(arg1cm,arg2cm,resultcm,alpha);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,alpha,1.5);
	}
}

BOOST_AUTO_TEST_SUITE_END()
