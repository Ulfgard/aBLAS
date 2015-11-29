/*!
 * 
 *
 * \brief       Basic implementation of a scheduler using a dependency graph
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
#ifndef ABLAS_SCHEDULING_SCHEDULING_HPP
#define ABLAS_SCHEDULING_SCHEDULING_HPP

#include <list>
#include <memory>
#include <functional>
#include <algorithm>
#include <atomic>
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

namespace aBLAS{ namespace scheduling{

class dependency_node;
class dependency_scheduling{
private:
	struct work_item{
		std::function<void()> workload;//the work to perform
		std::vector<typename std::list<work_item>::iterator > out_edges;//edges to work_items depending on this
		std::vector<dependency_node*> in_variables;//edges to used variables
		unsigned int active_dependencies;//number of dependencies this work_item is depending on before it can be computed
	};
	friend class dependency_node;
	std::size_t num_work_items(){
		boost::unique_lock<boost::mutex> lock(m_work_items_mutex);
		return m_work_items.size();
	}
public:
	void wait(){
		//block until all work is done
		while(num_work_items())
			boost::this_thread::yield();
	}
	~dependency_scheduling(){
		wait();
	}
	//function which writes to one variable
	void spawn(std::function<void()> && f, dependency_node& write_variable){
		enqueue_work(std::move(f),write_variable,{});
	}
	void spawn(std::function<void()> && f, dependency_node& write_variable, std::vector<dependency_node*>const& read_variables){
		enqueue_work(std::move(f),write_variable,read_variables);
	}
	//function which writes to one variable and reads one
	void spawn(std::function<void()> && f, dependency_node& write_variable,  dependency_node& read_variable){
		enqueue_work(std::move(f),write_variable,{&read_variable});
	}
	//function which writes to one variable and reads two
	void spawn(std::function<void()> && f, dependency_node& write_variable,  dependency_node& read_variable1, dependency_node& read_variable2 ){
		enqueue_work(std::move(f),write_variable,{&read_variable1, &read_variable2});
	}
	
	/// \brief Creates a closure filled with a temporary variable that survives until all kernels spawned in the closure are computed
	///
	/// Creates internally a temporary variable of type T and then calls work_item_producer synchronously with the temporary as argument.
	/// The work_item_producer spawns work items involving T. It is guaranteed that the temporary outlives the last kernel using it spawned this way.
	///  The only requirement on T is that it offers a method m_dependencies returning a reference to a dependency_node
	template<class T, class F>
	void create_closure(T&& temporary,F const& work_item_producer){
		//create variable and append kernels to it
		std::shared_ptr<T> temporary_copy(new T(std::move(temporary)));//unique_ptr would be ideal but does not work due to std::function requires the arguments to be copyable
		//let f add kernels to the temporary
		work_item_producer(*temporary_copy);
		//add the clean-up kernel
		spawn(std::function<void()>([temporary_copy](){/*call dtor of copy of ptr*/}),temporary_copy->dependencies());
	}
	template<class T1, class T2, class F>
	void create_closure(T1&& temporary1, T2&& temporary2,F const& work_item_producer){
		//create variable and append kernels to it
		std::shared_ptr<T1> temporary_copy1(new T1(std::move(temporary1)));
		std::shared_ptr<T2> temporary_copy2(new T2(std::move(temporary2)));
		//let f add kernels to the temporary
		work_item_producer(*temporary_copy1, *temporary_copy2);
		//add the clean-up kernel
		spawn(std::function<void()>([temporary_copy1, temporary_copy2](){/*call dtor of copy of both ptrs*/}),temporary_copy1->m_dependencies,{temporary_copy2->m_dependencies});
	}
	
	template<class T>
	void make_closure_variable(T&& temporary){
		create_closure(std::move(temporary),[](T&){});
	}
private:
	static void work_executor(dependency_scheduling& scheduler,std::list<work_item>::iterator work){
		//calculate workload
		work->workload();
		
		//signal scheduler that the work has been computed
		scheduler.finalize_work(work);
	}
	
	void submit(std::list<work_item>::iterator work){
		m_pool.submit(std::bind(work_executor,std::ref(*this), work));
	}
	
	/// \brief Adds a new work item to the graph and submits it directly if possible
	void enqueue_work(std::function<void()> && f, dependency_node& write_variable, std::vector<dependency_node*> const& read_variables);
	
	/// \brief Removes a finished work item from the dependency graph and submits work items that are now ready for execution
	void finalize_work(std::list<work_item>::iterator work);
	
	boost::basic_thread_pool m_pool;
	boost::mutex m_work_items_mutex;
	std::list<work_item> m_work_items;

};

class dependency_node{
private:
	friend class dependency_scheduling;
public:
	dependency_node():m_is_write_dependency(false){}
	bool is_ready(){
		return m_num_dependencies.load() == 0;
	}
	
	void wait(){
		while(!is_ready())
			boost::this_thread::yield();
	}
private:
	std::vector<dependency_scheduling::work_item*> m_dependencies;
	std::atomic_uint m_num_dependencies;
	bool m_is_write_dependency;

	//internal functions called for dependency management
	//all these functions can only be called sequentially. This means that the scheduler must be locked and there is only one scheduler!

	void write_dependency(dependency_scheduling::work_item* work){
		//write dependencies overwrite everything as work items will wait for all reads and writes to the same variables are enqueued sequentially
		m_dependencies.clear();
		m_dependencies.push_back(work);
		m_is_write_dependency = true;
		m_num_dependencies.store(1);
	}
	void add_read_dependency(dependency_scheduling::work_item* work){
		if(m_is_write_dependency){
			m_is_write_dependency = false;
			m_dependencies.clear();
			m_num_dependencies.store(0);
		}
		m_dependencies.push_back(work);
		++m_num_dependencies;
	}
	//remove finished dependencies in case they are still stored
	void remove_dependency(dependency_scheduling::work_item* work){
		std::vector<dependency_scheduling::work_item*>::iterator pos = std::find(m_dependencies.begin(),m_dependencies.end(),work);
		if(pos != m_dependencies.end()){
			--m_num_dependencies;
			m_dependencies.erase(pos);
		}
		if(m_dependencies.empty())
			m_is_write_dependency = false;
	}

};

void dependency_scheduling::enqueue_work(std::function<void()> && f, dependency_node& write_variable, std::vector<dependency_node*> const& read_variables){
	//do not allow any changes of andy work item while we collect information and change the structure
	boost::unique_lock<boost::mutex> lock(m_work_items_mutex);
	
	//collect all work items this work item has to wait for. these are write dependencies in 
	// read_variables (read a variable only after all previous write) and 
	// all dependencies of write_variable (only write when no-one else is using it)
	std::vector<work_item*> dependencies;
	dependencies.insert(dependencies.end(),write_variable.m_dependencies.begin(),write_variable.m_dependencies.end());
	for(dependency_node* node : read_variables){
		if(node->m_is_write_dependency)
			dependencies.push_back(node->m_dependencies.front());//there can only be one write dependency
	}
	//erase duplicates(todo: are duplicates possible at all?)
	dependencies.erase(std::unique(dependencies.begin(),dependencies.end()),dependencies.end());
	
	//construct work item
	work_item new_item;
	new_item.workload = std::move(f);
	new_item.in_variables = std::move(read_variables);
	new_item.active_dependencies = dependencies.size();
	m_work_items.push_back(std::move(new_item));
	
	//insert the work item into the graph
		
	//first add new dependency to all work items for which the new work item has to wait
	std::list<work_item>::iterator pos = std::prev(m_work_items.end());
	for(work_item* item : dependencies)
		item->out_edges.push_back(pos);
	
	//then add this kernel as read dependency to the enqueued variables
	for(dependency_node* node : pos->in_variables)
		node->add_read_dependency(&(*pos));
	
	//and also add write dependency to dependency list
	//this order ensures that write_dependencies are always
	//active even if the same variable is a read and write dependency
	write_variable.write_dependency(&(*pos));
	pos->in_variables.push_back(&write_variable);

	//submit this work item directly if it depends on nothing
	if(dependencies.empty()){
		submit(pos);
	}
}

void dependency_scheduling::finalize_work(std::list<work_item>::iterator work){
	boost::unique_lock<boost::mutex> lock(m_work_items_mutex);
	
	//mark dependencies as resolved and submit their work package to the queue
	//if all dependencies are resolved
	for(auto& item : work->out_edges){
		--item->active_dependencies;
		if(item->active_dependencies == 0){
			submit(item);
		}
	}
	
	//remove dependency from variable
	for(dependency_node* variable: work->in_variables){
		variable->remove_dependency(&(*work));
	}
	m_work_items.erase(work);
}


}

namespace system{
	scheduling::dependency_scheduling& scheduler(){
		static scheduling::dependency_scheduling scheduler;
		return scheduler;
	}
}

}

#endif