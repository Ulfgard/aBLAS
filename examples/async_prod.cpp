#include <aBLAS/matrix.hpp>
#include <aBLAS/matrix_expression.hpp>


int main(){
	aBLAS::matrix<double,aBLAS::row_major> r(100,100,0);
	aBLAS::matrix<double,aBLAS::row_major> x(100,100,1);
	for(std::size_t i = 0; i != 20; ++i){
		aBLAS::matrix<double,aBLAS::row_major> y(100,100,0.01);
		//enqueue some simple kernels
		//serial as r is written to in all kernels
		//~ noalias(r) += 2*x+1 + prod(x,y);
		//async version where async introduces intermediate variables.
		//as after the async computations all elements can be computed
		//elementwise, this also leads to a more efficient assignment
		//to r as r(i,j)= 2*x(i,j)+1+ t(i,j) where t=prod(x,y) is the async temporary
		noalias(r) += 2*x+1 + async(prod(x,y));
		//even though y is destroyed here, this is non-blocking as y is moved inside a
		//closure provided by the scheduler until all kernels are done
	}
	//asynchronous print-out
	//we enqueue a kernel that prints out the first value of r as soon as r is computed.
	//r might already be destroyed when the kernel is called and thus we can not store
	//a reference to r. But we can use its closure_types!
	aBLAS::matrix<double>::const_closure_type r_closure(r);
	aBLAS::system::scheduler().spawn([r_closure](){
		std::cout<<"expected 80, got: "<<r_closure(0,0)<<std::endl;
	},r.dependencies());//tell the scheduler which computations this kernel depends on
	std::cout<<"waiting for computations"<<std::endl;
	//destructor of the scheduler now blocks until everything is computed
	//somewhere here comes the asynchronous print-out...
}