#define BOOST_TEST_MODULE aBLAS_gemv
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <aBLAS/matrix.hpp>
#include <aBLAS/vector.hpp>
#include <aBLAS/kernels/gemv.hpp>

using namespace aBLAS;

//we test using the textbook definition.
template<class M, class V, class Result>
void checkMatrixVectorMultiply(M const& arg1, V const& arg2, Result const& result, double factor, double init = 0){
	BOOST_REQUIRE_EQUAL(arg1.size1(), result.size());
	BOOST_REQUIRE_EQUAL(arg2.size(), arg1.size2());
	
	for(std::size_t i = 0; i != arg1.size1(); ++i){
		double test_result = init;
		for(std::size_t k = 0; k != arg1.size2(); ++k){
			test_result += factor * arg1(i,k)*arg2(k);
		}
		BOOST_CHECK_CLOSE(result(i), test_result,1.e-10);
	}
}

BOOST_AUTO_TEST_SUITE (aBLAS_gemm)

BOOST_AUTO_TEST_CASE( aBLAS_gemv_dense_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	matrix<double,row_major> arg1rm(rows,columns);
	matrix<double,column_major> arg1cm(rows,columns);
	matrix<double,row_major> arg1rmt(columns,rows);
	matrix<double,column_major> arg1cmt(columns,rows);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*columns+0.2*j;
			arg1rmt(j,i) = arg1cmt(j,i) = i*columns+0.2*j;
		}
	}
	vector<double> arg2(columns);
	for(std::size_t j = 0; j != columns; ++j){
		arg2(j)  = 1.5*j+2;
	}
	double alpha=alpha;
	std::cout<<"\nchecking dense-dense gemv"<<std::endl;
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		kernels::gemv(arg1rm,arg2,result,alpha);
		checkMatrixVectorMultiply(arg1rm,arg2,result,alpha,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		kernels::gemv(arg1cm,arg2,result,alpha);
		checkMatrixVectorMultiply(arg1cm,arg2,result,alpha,1.5);
	}
}

BOOST_AUTO_TEST_SUITE_END()
