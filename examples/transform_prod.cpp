#include <aBLAS/matrix.hpp>
#include <aBLAS/matrix_expression.hpp>

int main(){
	typedef aBLAS::matrix<double,aBLAS::row_major> matrix;
	std::vector<matrix> r(1000,matrix(200,200,1));
	matrix x(200,200,1);
	
	//automatically parallel!
	std::transform(r.begin(),r.end(),r.begin(),[&x](matrix& m){
		return prod(m,x);
	});
	std::cout<<"waiting for computations"<<std::endl;
	aBLAS::system::scheduler().wait();//currently missing wait_for_all etc
	std::cout<<"expected 200, got:" <<r[99](0,0)<<std::endl;
}