#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <cblas.h>
#include "CSRMatrix.h"

template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate) :Matrix<T>(rows, cols, false), nnzs(nnzs)
{
	this->preallocated = preallocate;

	if (this->preallocated)
	{
		this->values.reset(new T[this->nnzs]);
		this->row_position.reset(new int[this->rows + 1]);
		this->col_index.reset(new int[this->nnzs]);
		this->row_index.reset(new int[this->nnzs]);
	}
}

template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T* values_ptr, int* row_position, int* col_index) :
	Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_position), col_index(col_index)
{}

// additional constructor if shared pointers are used
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, std::shared_ptr<T[]> values_ptr,
 						std::shared_ptr<int[]> row_position, std::shared_ptr<int[]>col_index) :
	Matrix<T>(rows, cols, values_ptr.get()), nnzs(nnzs), row_position(row_position), col_index(col_index)
{}


// No need to delete anything since using smart pointers
template <class T>
CSRMatrix<T>::~CSRMatrix()
{}


template <class T>
void CSRMatrix<T>::printMatrix() //overwrite the old print, since size changed
{
	std::cout << "\nPrinting sparse matrix";

	//loop over all arrays and print out
	std::cout << "\nrow_position: ";
	for (int i = 0; i < this->rows + 1; i++)
	{
		std::cout << this->row_position[i] << " ";
	}
	std::cout << "\ncol_index: ";
	for (int i = 0; i < this->nnzs; i++)
	{
		std::cout << this->col_index[i] << " ";
	}
	std::cout << "\nvalues: ";
	for (int i = 0; i < this->nnzs; i++)
	{
		std::cout << this->values[i] << " ";
	}
	std::cout << "\n";

}


// Sparse2dense method
template <class T>
Matrix<T> CSRMatrix<T>::sparse2dense() {
	auto* dense_mat = new Matrix<T>(this->rows, this->cols, true);
	for (int i = 0; i < dense_mat->rows; i++) {
		for (int j = 0; j < dense_mat->cols; j++) {
			dense_mat->values[j + i * (dense_mat->cols)] = 0;
		}
	}
	int row_index = 0;
	for (int i = 0; i < this->nnzs; i++) {
		if (i == this->row_position[row_index + 1])
			row_index++;
		dense_mat->values[this->col_index[i] + row_index * (dense_mat->cols)] = this->values[i];
	}
	return *dense_mat;
}


// CSRMatrix Version of matVecMult
template <class T>
void CSRMatrix<T>::matVecMult(T* input, T* output)
{
	for (int i = 0; i < this->rows; i++)
		output[i] = 0.0;

	// loop over rows
	for (int i = 0; i < this->rows; i++)
		for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
			output[i] += this->values[val_index] * input[this->col_index[val_index]];
}


// solver method contains our general solving methods for CSR Matrices
template <class T>
void CSRMatrix<T>::solver(const T* vec_b, T* vec_x, std::string method, int max_iter, double tol ) {
	if (method == "jacobi")
       this->jacobi(vec_b, vec_x, max_iter, tol);    
	else if (method == "gauss")
		this->gauss(vec_b, vec_x, max_iter, tol);  
	else if (method == "cg")
		this->cg(vec_b, vec_x, max_iter, tol);  
    else if (method == "cholesky")
        this->chol(vec_b, vec_x);

}

/*-----------------------------------------------------------------------------------------------*/
// Sparse Jacobi  Iterative Solver
//  Source: ACSE-3 Lecture 3 Notes, by Matthew Piggott. 
/*-----------------------------------------------------------------------------------------------*/
template <class T>
void CSRMatrix<T>::jacobi(const T* b, T* x, int max_iter, double tol)
{
	//check dimension
	if (this->cols != this->rows) {
		std::cerr << "input dimensions dont match" << std::endl;
		return;
	}

	// initialize output vector = zeros
	for (int i = 0; i < this->rows; i++)
		x[i] = 0;

	// Solve x from Ax = b using Gauss-Seidal
	double residual = 10;   // initial residual, will be computed in the while loop
	double sum = 0;

	for (int n = 0; n < max_iter; n++) {

		for (int i = 0; i < this->rows; i++) {
			sum = 0; // sum of Aij and Xj, except at i=j

			// for (int j = 0; j < this->cols; j++)
			for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
				if (this->col_index[val_index] != i)  // i < j or i > j
					sum += this->values[val_index] * x[this->col_index[val_index]];

			// x[i] = (b[i] - Sum A[ij]X[j])/A[ii]
			double Aii = 1;

			// try to extract Aii
			if (this->row_position[i] != this->row_position[i + 1]) //row is not empty
				for (int k = this->row_position[i]; k < this->row_position[i + 1]; k++)
					if (this->col_index[k] == i)
						Aii = this->values[k]; // diagonal element

			x[i] = (b[i] - sum) / (Aii);

		}

		// Calculate residual = ||A*x-b||, first by computing ax := A*x
		std::unique_ptr<double[]> ax(new double[this->rows]);		
		this->matVecMult(x, ax.get());

		// Set ax:= b-ax, then residual is the 2norm
		cblas_daxpby(this->rows, 1, b, 1, -1, ax.get(), 1);
		residual = cblas_dnrm2(this->rows, ax.get(), 1);

		if (residual < tol)
			break;
	}

	// Send error message that our solver did not converge properly
	if (residual >= tol)
		std::cerr << "ERROR! Jacobi iteration did not converge under tolerance" << std::endl;

}

/*-----------------------------------------------------------------------------------------------*/
// Sparse Gauss-Seidel Iterative Solver
//  Source: ACSE-3 Lecture 3 Notes, by Matthew Piggott. 
/*-----------------------------------------------------------------------------------------------*/
template <class T>
void CSRMatrix<T>::gauss(const T* b, T* x, int max_iter, double tol)
{
	// check dimension
	if (this->cols != this->rows) {
		std::cerr << "Improper input dimension" << std::endl;
		return;
	}

	// initialize output vector x as zeros
	for (int i = 0; i < this->rows; i++)
		x[i] = 0;

	// Create and initialize x_new
	std::unique_ptr<double[]> x_new(new double[this->rows]);
	for (int i = 0; i < this->rows; i++)
		x_new.get()[i] = 0;

	// intialize residual and sum
	double residual = 10;
	double sum = 0;
	double aii = 0;

	for (int n = 0; n < max_iter; n++) {
		for (int i = 0; i < this->rows; i++) {

			sum = 0;
			// for (int j = 0; j < this->cols; j++)
			for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++){
				if (this->col_index[val_index] < i)  //if j < i, sum += a_ij*x_new_j
					sum += this->values[val_index] * x_new.get()[this->col_index[val_index]];

				if (this->col_index[val_index] > i)  //if j > i, sum += a_ij*x_j
					sum += this->values[val_index] * x[this->col_index[val_index]];
			}

			aii = 0;
			// try to extract A_ii
			if (this->row_position[i] != this->row_position[i + 1]) //row is not empty
				for (int k = this->row_position[i]; k < this->row_position[i + 1]; k++) 
					if (this->col_index[k] == i) 
						aii = this->values[k];

			/* x_new[i] = (b[i] - sum)/A[ii] */
			x_new.get()[i] = (b[i] - sum) / (aii);
		}

		// Calculate residual = ||A*x-b||, first by computing ax := A*x
		std::unique_ptr<double[]> ax(new double[this->rows]);		
		this->matVecMult(x, ax.get());

		// Set ax:= b-ax, then residual is the 2norm
		cblas_daxpby(this->rows, 1, b, 1, -1, ax.get(), 1);
		residual = cblas_dnrm2(this->rows, ax.get(), 1);

		// Set x = x_new
		cblas_daxpby(this->rows, 1, x_new.get(), 1, 0, x, 1);

		if (residual < tol)
			return;
	}

	// Send error message that our solver did not converge properly
	if (residual >= tol)
		std::cerr << "ERROR! Gauss-Siedel iteration did not converge under tolerance" << std::endl;
}


/*-----------------------------------------------------------------------------------------------*/
// Sparse Conjugate Gradient Iterative Solver
//  Source: ACSE-3 Lecture 3 Notes, by Matthew Piggott. 
/*-----------------------------------------------------------------------------------------------*/
template <class T>
void CSRMatrix<T>::cg(const T* b, T* x, int max_iter, double tol)
{
	std::shared_ptr<double[]> residual(new double[this->rows]);
	std::shared_ptr<double[]> p(new double[this->rows]);
	std::shared_ptr<double[]> Ap(new double[this->rows]);

	double sum_r = 0;
	double sum_r_new = 0;
	double alpha = 0;

	for (int i = 0; i < this->rows; i++){
		x[i] = 0;
		residual[i] = b[i];
		p[i] = b[i];
		Ap[i] = 0;
	}

	sum_r = cblas_dnrm2(this->rows, residual.get(), 1);
	sum_r *= sum_r;

	for (int n = 0; n < max_iter; n++){

		// Ap := A*p
		this->matVecMult(p.get(), Ap.get());

		// alpha := sum_r/p^TAp
		alpha = sum_r / cblas_ddot(this->rows, Ap.get(), 1, p.get(), 1);

		// x := x + alpha*p
		cblas_daxpby(this->rows, alpha, p.get(), 1, 1, x, 1);
	
		// r := r - alpha*w
		cblas_daxpby(this->rows, -alpha, Ap.get(), 1, 1, residual.get(), 1);

		// sum_r_new = ||r_new||^2
		sum_r_new = cblas_dnrm2(this->rows, residual.get(), 1);
		sum_r_new *= sum_r_new;

		if (sqrt(sum_r_new) < tol)
			return;

		// p := r + (sum_r_new/sum_r)*p
		for (int i = 0; i < this->rows; i++)
			p[i] = residual[i] + (sum_r_new / sum_r) * p[i];

		sum_r = sum_r_new;
	
	}
}


/*-----------------------------------------------------------------------------------------------*/
// Cholesky factorisation solver: Solves Ax = b by computing A = LL^T
// Source:  Parallel Numerical Algorithms, pp. 55-90 D. E. Keyes, A. Sameh, V. Venkatakrishnan.
// N.B. We assume input matrix, A, being symmetric, only has lower triangular values stored
// 
// Method: 
// 1. Compute elmination tree, etree, of our sparse lower triangular matrix A
// 2. Symbolic Factorisation: Get the row positions and column indices of cholesky factor L using etree
// 3. Numerical Factorisation: Caclulate the actual non-zero values of L with the usual Cholesky method
//	This consists of iterating through the columns and doing: 
//	3.1 cmod(k,j): subtraction of column k from column j; k < j
//	3.2 cdiv(j): calculation of L_ii = sqrt(A_ii), and dividing the rest of column by this 
// 4. Forward substitution solving for y: Ly = b
// 5. Backward substitution solving for x: L^Tx = y
/*-----------------------------------------------------------------------------------------------*/
template <class T>
void CSRMatrix<T>::chol(const T* b, T* x){

	//  STEP 1. Calculate Elimination Tree, etree

	// Initialize etree
	unique_ptr<int[]> etree(new int[this->rows]);
	// for (int i = 0; i < this->rows; i++)
	// 	etree.get()[i] = this->rows;

 	// Variable used to keep track of row of a variable throughout
 	int r = 0; 

	// Calculate etree, algorithm by Liu
	for (int i = 0; i < this->rows; i++){
		etree.get()[i] = -1;

		// for all x_j in Adj(x_i), j < i
		for (int j = this->row_position[i]; j < this->row_position[i+1]-1; j++){

			// find the root x_r of the tree in the forest containing the node x_j
			r = this->col_index[j];

			while (etree.get()[r] != -1 && etree.get()[r] != i)
				r = etree.get()[r];

			if (etree.get()[r] == -1)
				etree.get()[r] = i;
		}

	}


	// STEP 2. Symbolic Factorisation using etree 

	// We initialize vectors of L's col_index, values and row_position
	// so we can easily insert new values 
	std::vector<int> l_col_index; 
	for (int i = 0; i < this->nnzs; i++)
		l_col_index.push_back(this->col_index[i]);

	std::vector<double> l_values; 
	for (int i = 0; i < this->nnzs; i++)
		l_values.push_back(this->values[i]);

	std::vector<int> l_row_position; 
	for (int i = 0; i < this->rows+1; i++)
		l_row_position.push_back(this->row_position[i]);

	// A set existing_cols is used to keep track of all current columns in a row
	// to avoid dupulicating the insertion of a new column.
	set<int> existing_cols;

	// counter keeps track of the number of insertions overall (so when making a new 
	// insertion in vector, we take into account the fact that the vector has
	// already grown by counter). 
	int counter = 0;

	// col_in_row_k keeps track of current column being traversed in elimination tree
	int col_in_row_k;

	// We now construct L's row_pos and col_index from the elimination tree
	for (int k = 0; k < this->rows; k++){
		existing_cols.clear();

		for (int j = this->row_position[k]; j < this->row_position[k+1]; j++)
			existing_cols.insert(this->col_index[j]);

		for (int j = this->row_position[k]; j < this->row_position[k+1] - 1; j++){
			col_in_row_k = this->col_index[j];

			// Stop when our column is past the current working columm.
			while (col_in_row_k < k){

				// if we've found a new column, add it! 
				if (existing_cols.find(col_in_row_k) == existing_cols.end()){

					// insert col_index, and a corresponding 0 value for L
					l_col_index.insert(l_col_index.begin() + j + counter, col_in_row_k);
					l_values.insert(l_values.begin() + j + counter, 0);

					// update row_pos (increment all row pos's ahead)
					for (int kk = k+1; kk < this->rows+1; kk++)
						l_row_position[kk] ++;

					// add this new col to existing cols
					existing_cols.insert(col_in_row_k);	

					counter++; // used to shift current col_index vector for future insertions
				}

				// Go up evaluation tree
				col_in_row_k = etree.get()[col_in_row_k];
			}
		}
	}


	// new number of non-zero entries in cholesky factor L 
	int chol_nnzs = l_row_position.back(); 


	// STEP 3. Numerical Cholesky Decomposition
	// diag keeps track of whether we have got the diagonal yet (the first element of column) 
	// and if so, diag_pos holds its position for the cdiv(j) step.
	bool diag = true; 
	int diag_pos = 0;

	for (int j = 0; j < this->cols; j++){
		
		int k = 0;
		// cmod(j,k): subtraction of column k from column j; k < j
		for (int kk = l_row_position[j]; kk < l_row_position[j+1] - 1; kk++){ // -1 ensures we ignore diagonal
			// Loop occurs for all k in Struct(L_j*) = the set of non zero columns in row j
			k = l_col_index[kk];

			// Go through all the elements l_*j in column j (always below the j-1 diagonal in a lower tri matrix)
			for (int i = l_row_position[j]; i < chol_nnzs; i++){
				if (l_col_index[i] == j){

					// find the row, i, l_*j is in
					r = 0; 
					while (l_row_position[r] <= i)
						r++; 	
					
					// find a_ik		
					for (int z = l_row_position[r-1]; z < l_row_position[r]; z++)
						if (l_col_index[z] == k)
							// if we've got a_ik, find a_jk
							for (int zz = l_row_position[j]; zz < l_row_position[j+1]; zz++)
								if (l_col_index[zz] == k){
									// l_ij = a_ij - a_ik*a_jk
									l_values[i] -= l_values[z]*l_values[zz]; 
									goto break_out;
								}
					
					// we use goto statement to break out of nested for loop above
					break_out: continue;
					
				}
			}

		}

		// cdiv(j): calculation of L_ii = sqrt(A_ii), and dividing the rest of column by this 

		// boolean keeps track of the first element in the column (which is the diagonal) to sqrt
		diag = true;

		// we start at l_row_position[j] as the submatrix does contain anything from column j
		for (int i = l_row_position[j]; i < chol_nnzs; i++){
			if (l_col_index[i] == j){
				if (diag){
					l_values[i] = sqrt(l_values[i]); // a_ii = sqrt(a_ii)
					diag = false; // no more diagonals in this column!
					diag_pos = i;
				}
				else l_values[i] /= l_values[diag_pos];
			}
		}
		
	}

	// STEP 4. Forward Substitution Solve for y: Ly = b
	double sum = 0;
	unique_ptr<double> y(new double[this->rows]); 

	for (int i = 0; i < this->rows; i++){
		sum = 0;
		for (int z = l_row_position[i]; z < l_row_position[i+1] - 1; z++){ // -1 ensures we avoid diagonal
			// add L_ij*y[j] to sum
			sum += l_values[z]*y.get()[l_col_index[z]];
		}
		y.get()[i] = (b[i] - sum)/(l_values[l_row_position[i+1]-1]); // y_i = (b_i - sum)/L_ii
	}


	//  STEP 5. Backward Substitution Solve for x: L^Tx = y
	for (int i = this->rows-1; i >= 0; i--){
		sum = 0;
		for (int z = l_row_position[i+1]; z < chol_nnzs; z++){
			// add L_ij*y[j] to sum
			if (l_col_index[z] == i){
				r = 0; 
				// find the row just after of our element z
				while (l_row_position[r] <= z)
					r++; 
				sum += l_values[z]*x[r-1];
			}
		}
		x[i] = (y.get()[i] - sum)/(l_values[l_row_position[i+1]-1]); // x_i = (y_i - sum)/L_ii
	}

}