/*!
 * 
 *
 * \brief       Basic error checks
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
#ifndef ABLAS_DETAIL_EXCEPTION_H
#define ABLAS_DETAIL_EXCEPTION_H

#include <string>
#include <exception>

namespace aBLAS {

	/**
	* \brief Top-level exception class of the shark library.
	*/
	class Exception : public std::exception {
	public:
		/**
		* \brief Default c'tor.
		* \param [in] what String that describes the exception.
		* \param [in] file Filename the function that has thrown the exception resides in.
		* \param [in] line Line of file that has thrown the exception.
		*/
		Exception( const std::string & what = std::string(), const std::string & file = std::string(), unsigned int line = 0 ) : m_what( what ),
			m_file( file ),
			m_line( line ) {
		}

		/**
		* \brief Default d'tor.
		*/
		~Exception( ) throw() {}

		/**
		* \brief Accesses the description of the exception.
		*/
		inline const char* what() const throw() {
			return m_what.c_str();
		}

		/**
		* \brief Accesses the name of the file the exception occurred in.
		*/
		inline const std::string & file() const {
			return( m_file );
		}

		/**
		* \brief Accesses the line of the file the exception occured in.
		*/
		inline unsigned int line() const {
			return( m_line );
		}

	protected:
		std::string m_what; ///< Description of the exception.
		std::string m_file; ///< File name the exception occurred in.
		unsigned int m_line; ///< Line of file the exception occurred in.
	};

}

/**
* \brief Convenience macro that creates an instance of class aBLAS::exception,
* injecting file and line information automatically.
*/
#define ABLASEXCEPTION(message) aBLAS::Exception(message, __FILE__, __LINE__)

inline void THROW_IF(bool unexpectedCondition, const std::string& message)
{
	if (unexpectedCondition)
		throw ABLASEXCEPTION(message);
}

// some handy macros for special types of checks,
// throwing standard error messages
#ifndef NDEBUG
#define ABLAS_RANGE_CHECK(cond) do { if (!(cond)) throw ABLASEXCEPTION("range check error: "#cond); } while (false)
#define ABLAS_SIZE_CHECK(cond) do { if (!(cond)) throw ABLASEXCEPTION("size mismatch: "#cond); } while (false)
#else
#define ABLAS_RANGE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#define ABLAS_SIZE_CHECK(cond) do { (void)sizeof(cond); } while (false)
#endif

#endif
