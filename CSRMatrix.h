/*
 * File: CSRMatrix.h
 * --------------
 * This interface exports the CSR Matrix class which is a sub class from Matrix class
 */

#pragma once
#include <vector>
#include <memory>
#include "Matrix.h"
#include <string>

using namespace std;

template <class T>
class CSRMatrix : public Matrix<T>
{
public:

	// constructor where we want to preallocate memory, own its own memory
	CSRMatrix(int rows, int cols, int nnzs, bool preallocate);
	CSRMatrix(int rows, int cols, int nnzs, T* values_ptr, int* row_position, int* col_index);

	// additional constructor if shared pointers are used
	CSRMatrix(int rows, int cols, int nnzs, std::shared_ptr<T[]> values_ptr, 
	std::shared_ptr<int[]> row_position, std::shared_ptr<int[]>col_index);

	//destructor
	~CSRMatrix();  

	// print the values of matrix
	virtual void printMatrix();

	/*
	* Method: Sparse to dense matrix
	* Useage: mat_A.sparse2dense()
	* -------------------------------------------
	* Returns a dense matrix filled with the values
	* of the sparse input. Useful for debugging other CSR methods!
	*/
	Matrix<T> sparse2dense();


	/*
	* Method: solver for sparse
	* Useage: sparse_mat_A.solver(x, b)
	* ---------------------------
	* Sets the matrix x to be the solution to the linear system Ax = b in CSR matrix format
	*/
	void solver(const T* vec_b, T* vec_x, std::string method = "gauss", int max_iter = 5000, double tol = 1e-5);


	/*
	* Method: matVectMult
	* Useage: mat_A.matMatMult(vect_b, vect_c);
	* ----------------------------------------
	* Sets vector vect_c to be the matrix-vector product mat_A*vect_b
	*/
	void matVecMult(T* input, T* output);


	/*
	* Method: Iterative Sparse Solvers
	* Useage: mat_A.jacobi(&b, &x, 10000, 1e-5)
	* -------------------------------------------
	* Sets the matrix x to be the solution to the linear system Ax = b 
	* using iterative solvers below
	*/
	void cg(const T* b, T* x, int max_iter = 1000, double tol = 1e-5);
	void jacobi(const T* b, T* x, int max_iter = 5000, double tol = 1e-5);
	void gauss(const T* b, T* x, int max_iter = 5000, double tol = 1e-5);


	/*
	* Method: Cholesky Decomposition Solver
	* Useage: mat_A.chol(&b, &x)
	* -------------------------------------------
	* Sets the matrix x to be the solution to the linear system Ax = b 
	* using Cholesky decomposition of A and backwards substitution
	*/
	void chol(const T* b, T* x);


	// Variables
	std::shared_ptr<int[]> row_position{ nullptr };
	std::shared_ptr<int[]> col_index{ nullptr };
	std::shared_ptr<int[]> row_index{ nullptr };
	// no need 'values', already inherited T* values = nullptr;

	int nnzs = -1;  //number of non-zero elements

};