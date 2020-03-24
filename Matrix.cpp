#include <iostream>
#include <vector>
#include <cmath>
#include "Matrix.h"
#include <memory>
#include <iomanip>
#include <cblas.h>
#include <string> 

// constructor
//class :: function parameter from class (this function)
template <class T>
Matrix<T>::Matrix(int rows, int cols, bool preallocate) : 
rows(rows), cols(cols), size_of_values(rows* cols), preallocated(preallocate)
{
	// :rows(rows) eq to   this->rows = rows; in {}

	// if we want to handle memory ourselves
	if (this->preallocated)
	{
		this->values.reset( new T[this->size_of_values] );
	}
}

// constructor if given existing memory; just pass in values
template <class T>
Matrix<T>::Matrix(int rows, int cols, T *values_ptr): 
rows(rows), cols(cols), size_of_values(rows * cols), values(values_ptr)
{}


/* destructor: no need since values are owned by smart pointer */
template <class T>
Matrix<T>::~Matrix()
{}


template <class T>
void Matrix<T>::printValues(){
	std::cout << "Printing values " << std::endl;
	for (int i = 0; i < this->size_of_values; i++)std::cout << this->values.get()[i] << " ";
	std::cout << std::endl;
}


template <class T>
void Matrix<T>::printMatrix(){
	std::cout << std::endl;
	for (int j = 0; j < this->rows; j++)
	{
		for (int i = 0; i < this->cols; i++)
			// use row-major ordering here 
		{
			std::cout << std::setw(10) << this->values.get()[i + j * this->cols] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl <<std :: endl;
}


template <class T>
void Matrix<T>::matMatMult(Matrix<T>& mat_right, Matrix<T>& output)
{
	if (this->cols != mat_right.rows)
	{
		std::cerr << "Input dimensions for matrices don't match" << std::endl;
		return;
	}

   // Check if our output matrix has had space allocated to it
   if (output.values != nullptr) 
   {
      // Check our dimensions match
      if (this->rows != output.rows || this->cols != output.cols)
      {
         std::cerr << "Input dimensions for matrices don't match" << std::endl;
         return;
      }      
   }
   // The output hasn't been preallocated, so we are going to do that
   else
   {
      output.values.reset( new T[this->rows * mat_right.cols]);
      output.preallocated = true;
   }

	  // Set values to zero before hand
	for (int i = 0; i < output.size_of_values; i++)
		output.values[i] = 0;

// Now we can do our matrix-matrix multiplication
	for (int i = 0; i < this->rows; i++)
		for (int k = 0; k < this->cols; k++)
			for (int j = 0; j < mat_right.cols; j++)
				output.values[i * output.cols + j] 
				+= this->values[i * this->cols + k] * mat_right.values[k * mat_right.cols + j];
}


template <class T>
void Matrix<T>::matVecMult(T* vect_right, T* output)
{
	// multiply matrix to vector
	for (int i = 0; i < this->rows; i++)
		for (int j = 0; j < this->cols; j++)
		 output[i] += this->values[i * this->cols + j] * vect_right[j];
}



template <class T>
Matrix<T> Matrix<T>::operator+(Matrix<T>& mat_right){
    // Check rows and columns match in left and right matrices
    if (rows != mat_right.rows || cols != mat_right.cols){
        std::cerr << "Input dimensions for matrix addition don't match" << std::endl;
    }
	// create new matrix object to return
	Matrix mat_sum = Matrix(rows, cols, true);
	for (int i = 0; i < this->rows; i++){
		for (int j = 0; j < this-> cols; j++){
			mat_sum.values.get()[i * cols + j] 
			= this->values.get()[i * cols + j] + mat_right.values.get()[i * cols + j];
		}
	}
	return mat_sum;
}


/*-----------------------------------------------------------------------------------------------*/
// Solver function to solve linear system Ax = b with several methods. 

template <class T>
void Matrix<T>::solver(const T* vec_b, T* vec_x, std::string method, int max_iter, double tol){
	if (method == "LU")
		this->luSolve(vec_b, vec_x);
    else if (method == "jacobi")
        this->jacobi(vec_b, vec_x, max_iter, tol);    
	else if (method == "gauss")
		this->gauss(vec_b, vec_x, max_iter, tol);
	else if (method == "cg")
		this->cg(vec_b, vec_x, max_iter, tol);
    else if (method == "GMRES")
        this->gmres(vec_b, vec_x, max_iter, tol);    

}


/*-----------------------------------------------------------------------------------------------*/
// Dense Jacobi Iterative Solver
//  Source: ACSE-3 Lecture 3 Notes, by Matthew Piggott. 
/*-----------------------------------------------------------------------------------------------*/
template <class T>
void Matrix<T>::jacobi(const T* b, T* x, int max_iter, double tol)
{
	//check dimension
	if (this->cols != this->rows){
		std::cerr << "Improper input dimension" << std::endl;
		return;
	}

	// initialize output vector = zeros
	for (int i = 0; i < this->rows; i++){
		x[i] = 0;
	}

	double residual = 10;   // initial residual, will be computed in the while loop

	for (int n = 0; n < max_iter; n++){

		for (int i = 0; i < this->rows; i++){
			double sum = 0; // sum of Aij and Xj, except at i=j

			for (int j = 0; j < this->cols; j++)
			{
				if (i != j){
					sum += this->values[i * this->cols + j] * x[j];
				}
			}
			// x[i] = (b[i] - Sum A[ij]X[j])/A[ii]
			x[i] = (b[i] - sum) / (this->values[i * this->rows + i]);
		}

		// Calculate residual = ||A*x-b||, first by computing ax := A*x
		std::unique_ptr<double[]> ax(new double[this->rows]);		
		cblas_dgemv(CblasRowMajor, CblasNoTrans,
			this->rows, this->cols, 1, this->values.get(), this->cols, x, 1, 0, ax.get(), 1);

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
// Dense Gauss-Seidel Solver
// Source: ACSE-3 Lecture 3 Notes, by Matthew Piggott. 

template <class T>
void Matrix<T>::gauss(const T* b, T* x, int max_iter, double tol)
{
	// check dimension
	if (this->cols != this->rows)
	{
		std::cerr << "Improper input dimension" << std::endl;
		return;
	}

	// initialize output vector x as zeros
	for (int i = 0; i < this->rows; i++)
		x[i] = 0;

	// Create and initialize x_new
	std::unique_ptr<double[]> x_new(new double[rows]);
	for (int i = 0; i < this->rows; i++)
		x_new.get()[i] = 0;

	// intialize residual nad sum
	double residual = 10;
	double sum = 0;

	for (int n = 0; n < max_iter; n++) {
		for (int i = 0; i < this->rows; i++) {
			sum = 0;

			for (int j = 0; j < i; j++) {
				sum += this->values.get()[i * this->cols + j] * x_new.get()[j];
			}

			for (int j = i + 1; j < this->cols; j++)
				sum += this->values.get()[i * this->cols + j] * x[j];

			/* x_new[i] = (b[i] - sum)/A[ii] */
			x_new.get()[i] = (b[i] - sum) / (this->values.get()[i * this->rows + i]);
		}

		/*
		* Calculate residual = ||A*x-b||, first by computing ax := A*x
		* N.B. parameter 10 in dgemv call is 0 so no need to zero out ax when we initialize
		*/
		std::unique_ptr<double[]> ax(new double[this->rows]);
		cblas_dgemv(CblasRowMajor, CblasNoTrans,
			this->rows, this->cols, 1, this->values.get(), this->cols, x, 1, 0, ax.get(), 1);

		/* Set ax:= b-ax, then residual is the 2norm*/
		cblas_daxpby(this->rows, 1, b, 1, -1, ax.get(), 1);
		residual = cblas_dnrm2(this->rows, ax.get(), 1);

		/* Set x = x_new */
		cblas_daxpby(this->rows, 1, x_new.get(), 1, 0, x, 1);

		if (residual < tol)
			break;

	}

	// Send error message that our solver did not converge properly
	if (residual >= tol)
		std::cerr << "ERROR! Gauss-Siedel iteration did not converge under tolerance" << std::endl;
}


/*-----------------------------------------------------------------------------------------------*/
// Dense Conjugate Gradient Solver
// Source: ACSE-3 Lecture 3 Notes, by Matthew Piggott. 

template <class T>
void Matrix<T>::cg(const T* b, T* x, int max_iter, double tol)
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
    	cblas_dgemv(CblasRowMajor, CblasNoTrans,
			this->rows, this->cols, 1, this->values.get(), this->cols, p.get(), 1, 0, Ap.get(), 1);

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
// GMRES Solver! 
// Generalized Minimal Residual Method
// Algorithm from "Numerical Linear Algebra", by Trefethen and Bau (SIAM, 1997), pp. 266-270

template <class T>
void Matrix<T>::gmres(const T* b, T* x, int max_iter, double tol)
{	
	// make a copy of b, b_vec, since we're going to modify it
	std::shared_ptr<double[]> b_vec(new double[this->rows]);
	cblas_daxpby(this->rows, 1, b, 1, 0,  b_vec.get(), 1);

	// initialize Q, H matrices, b_norm 
	std::unique_ptr<Matrix<double>> Q(new Matrix(max_iter+1, rows, true)); // NOTE: We are using column order for Q
	std::unique_ptr<Matrix<double>> H(new Matrix(max_iter+1, max_iter, true)); 

	// Preconditioner: b := M^-1b, where M = diag(A)
	for (int i = 0; i < this->rows; i++)
	b_vec.get()[i] /= this->values.get()[i + this->rows*i];

	double b_norm = cblas_dnrm2(this->rows, b_vec.get(), 1); 
	for (int i = 0; i < Q->cols; i++)
	Q->values[i] = b_vec.get()[i]/b_norm; 

	// initialize v, to be used in Arnoldi iteration 
	std::unique_ptr<double[]> v(new double[rows]); 
	double v_norm = 0;

	// To store our previous Givens rotations from least squares, initialize G to be Identity 
	std::unique_ptr<Matrix<double>> G(new Matrix(max_iter+1, max_iter+1, true)); 
	for (int i = 0; i < G->size_of_values; i++)
	G->values.get()[i] = 0;
	for (int i = 0; i < G->rows; i++)
	G->values.get()[i + i*G->rows] = 1;

	// initialize least squares minimizer, ||Hy- ||b||*e_1||, to be used at end 
	std::shared_ptr<double[]> y(new double[max_iter]);
	std::shared_ptr<double[]> e_1(new double[max_iter+1]);
	for (int i = 0; i < max_iter+1; i++)
	e_1.get()[i] = 0;
	e_1.get()[0] = b_norm;
	double residue = 1;

	// Arnoldi iteration loop; we compute H, Q such that H = Q*AQ
	for (int n = 0; n < max_iter; n++){

		// Set v := Aq_n, raw pointer with pre-allocated memory
		for (int i = 0; i < this->rows; i++)
			v.get()[i] = 0;
		
		auto *qn = new Matrix(this->rows, 1, &(Q->values.get()[n*Q->cols])); 
		cblas_dgemv(CblasRowMajor, CblasNoTrans, 
					this->rows, this->cols, 1, this->values.get(), this->cols, qn->values.get(), 1, 1, v.get(), 1);
		// Preconditioner: v := M^-1v, where M = diag(A)
		for (int i = 0; i < this->rows; i++)
			v.get()[i] /= this->values.get()[i + this->rows*i];

		// Compute the next column values of H, H[0:n, n] 
		for (int j = 0; j < n + 1; j++){
			// qj points to j-th row of Q, raw pointer with pre-allocated memory
			auto *qj = new Matrix(this->rows, 1, &(Q->values.get()[j*Q->cols]) ); 
			// Set H[j, n] = H_jn := q_j.v 
			H->values.get()[n + j*H->cols] = cblas_ddot(rows, qj->values.get(), 1, v.get(), 1);
			// Set v = v - H_jn*q_j 
			cblas_daxpby(this->rows, -H->values.get()[n + j*H->cols], qj->values.get(), 1, 1,  v.get(), 1);
		}


		// Compute the final needed in H, H[n+1, n] = norm(v), and the next row of Q 
		v_norm = cblas_dnrm2(rows, v.get(), 1);
		H->values[(n+1)*H->cols + n] = v_norm;
		// Q[:, n+1] = q_{n+1} = v/norm(v) 
		for (int i = 0; i < Q->cols; i++)
			Q->values.get()[i + (n+1)*Q->cols] = v.get()[i]/v_norm;


		// Least squares section of GMRES
		// We solve min_y = \|Hy - e_1||, so then x = Q^Ty 
		// Loop: if residue from least squares is under tolerance, calculate x and return           
		if (H->leastSquares(y.get(), e_1.get(), *G, n+2) < tol || n == max_iter-1){ 
			// Note: Q is stored column-wise, so CblasTrans is passed 
			cblas_dgemv(CblasRowMajor, CblasTrans, 
						Q->rows, Q->cols, 1, Q->values.get(), Q->cols, y.get(), 1, 0, x, 1);

			if (n == max_iter-1)
				std::cerr << "Warning, max number of iterations occured" << std::endl;

			return;
		}
	}

}


/*-----------------------------------------------------------------------------------------------*/
// Least squares solver (using QR factorisation via Givens rotations)
// !!!!!!TODO!!!!!: Speed up Givens multiplations, only 2 rows change per multiplication
 
template <class T>
double Matrix<T>::leastSquares(T* x, T* b, Matrix<T> &G, int n){
    
    // R is a copy of our Hessenberg matrix
    // TODO: add copy constructor to Matrix Class 
    std::unique_ptr<Matrix<double>> R( new Matrix<double>(n, n-1, true) );
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            R->values[j + i*(n-1)] = this->values.get()[j + i*(this->cols)];

    // Set prior combined givens Q from values of G 
    std::unique_ptr<Matrix<double>> Q( new Matrix<double>(n, n, true) );
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Q->values.get()[i + j*n] = G.values.get()[i + j*G.rows]; 
    
    // multiply R, the copy of H_n, by Q, all the prior rotations 
    std::unique_ptr<Matrix<double>> mat_mul( new Matrix<double>(n, n-1, true) );
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                n, n-1, n, 1, Q->values.get(), n, R->values.get(), n-1, 0, mat_mul->values.get(), n-1);
    for (int i = 0; i < n*(n-1); i++)
        R->values.get()[i] = mat_mul->values.get()[i];

    // Initialize new givens to be identity 
    std::unique_ptr<Matrix<double>> givens( new Matrix<double>(n, n, true) );
    for (int i = 0; i < n*n; i++)
        givens->values.get()[i] = 0;
    for (int i = 0; i < n; i++)
        givens->values.get()[i + i*n] = 1;

    // The new givens will be the rotation G_k,k+1(theta) 
    int k = n-2;
    double x_val = R->values.get()[k*(n-1) + k]; 
    double y_val = R->values.get()[(k+1)*(n-1) + k]; 
    double r = sqrt(x_val*x_val + y_val*y_val);
    givens->values.get()[k*n + k] = x_val/r; 
    givens->values.get()[k*n+ k + 1] = y_val/r; 
    givens->values.get()[k*n + n + k] = -y_val/r;
    givens->values.get()[k*n + n+1 +k] = x_val/r;

    // Multiply latest givens roation with Q and store this in G 
    std::unique_ptr<Matrix<double>> giv_mul( new Matrix<double>(n, n, true) );
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                n, n, n, 1, Q->values.get(), n, givens->values.get(), n, 0, giv_mul->values.get(), n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            G.values.get()[i + j*G.rows] = giv_mul->values.get()[i + j*n]; 

    // d = Q^Tb, just the first row of Q*b_norm basically
    std::unique_ptr<double[]> d(new double[n]); 
    cblas_dgemv(CblasRowMajor, CblasTrans, n, n, 1, Q->values.get(), n, b, 1, 0, d.get(), 1);
    
    // backward substitution Ry = d  
    double sum = 0;
    for (int i = n-2; i >= 0; i--){
        sum = 0; 
        for (int j = n-2; j > i; j--)
            sum += x[j]*R->values.get()[j + i*(n-1)];
        x[i] = (d.get()[i]-sum)/R->values.get()[i + i*(n-1)];
    } 

    // Calculate and return residue 
    // 1. Set r = b
    // 2. Set R back to H_n
    // 3. Compute residue = ||(H_ny - ||b||e_1)||/||b||
     
    double residue = 1;
    std::unique_ptr<double[]> res(new double[n]); 
    for (int i = 0; i < n; i++)
        res.get()[i] = 0; 
    res.get()[0] = b[0];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n-1; j++)
            R->values[j + i*(n-1)] = this->values.get()[j + i*(this->cols)];

    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n-1, 1, R->values.get(), n-1, x, 1, -1, res.get(), 1);
    return cblas_dnrm2(n, res.get(), 1)/b[0];
 }
 


/*-----------------------------------------------------------------------------------------------*/
// LUP Decomposition Solver.
//	Source: ACSE-3 Lecture 3 Notes, by Matthew Piggott. 
//	1. Decompose Matrix A into PLU Matrix, where P is permutation matrix for pivoting row, 
//     L is the lower triangle component from Matrix A, and Matrix U is triangle component from 
//     Matrix A (in this case, Matrix U is reduced Matrix A).
//	2. Transpose Matrix P to be multiplied later with vector b
//	3. Solve y for L y = P^-1 b where y = U x using forward subtitution
//	4. Solve x for U x = u using backward subtitution
/*-----------------------------------------------------------------------------------------------*/
template <class T>
void Matrix<T>::luSolve(const T* b, T* x)
{
	// initialize L_matrix as part of PLU decomposition
	std::shared_ptr<Matrix<double>> L_tri(new Matrix<double>(this->rows, this->cols, true));
	for (int i = 0; i < L_tri->size_of_values; i++) 
		L_tri->values[i] = 0;

	// Intialize Permutation Matrix perm_mat as Identity
	std::shared_ptr<Matrix<double>> perm_mat(new Matrix<double>(this->rows, this->cols, true));
	for (int i = 0; i < perm_mat->size_of_values; i++) 
		perm_mat->values[i] = 0;
	for (int i = 0; i < cols; i++) 
		perm_mat->values[i * cols + i] = 1;

	// factorize A into L, U, and P matrix
	this->lupDecompose(L_tri.get(), perm_mat.get());

	// Set b_perm := P*b
	std::shared_ptr<double[]> b_perm(new double[this->rows]);
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
	 perm_mat->rows, perm_mat->cols, 1, perm_mat->values.get(), perm_mat->cols, b, 1, 0, b_perm.get(), 1);

	// Initialize  y for forward and backward substitution
	std::shared_ptr<double[]> y(new double[this->rows]);

	// solve y for L y = P^-1 b system prior calculating x
	L_tri->forwardSub(b_perm.get(), y.get());

	// solve y for L y = P^-1 b system prior calculating x
	this->backwardSub(y.get(), x);
}


template <class T>
void Matrix<T>::lupDecompose(Matrix<T>* L_tri, Matrix<T>* perm_mat)
{
	//Construct pivot parameter
	double max_row_val, s;
	int max_row;

	// CHECK FIRST FOR MATRIX DIMENSION

	// finding pivot row and perform decompostiion
	// i stands for whole A rows, k stands for rows below rows i
	for (int i = 0; i < this->cols; i++){
		max_row_val = (fabs(this->values[i + i * this->cols]));
		max_row = i + i * this->cols;
		for (int k = i + 1; k < this->rows; k++){
			if (fabs(this->values[i + k * this->cols]) > max_row_val){
				max_row_val = (fabs(this->values[i + k * this->cols]));
				max_row = i + k * this->cols;
			}
		}

		// perform row pivot if necessary (when maximum row is located below current pivot row)
		if (max_row != i + i * this->cols){
			for (int j = 0; j < this->cols; j++){
				std::unique_ptr<double[]> temp(new double[this->cols]);
				temp[i] = this->values[j + i * this->cols];
				this->values[j + i * this->cols] = this->values[j + max_row - i];
				this->values[j + max_row - i] = temp[i];

				temp[i] = L_tri->values[j + i * L_tri->cols];
				L_tri->values[j + i * L_tri->cols] = L_tri->values[j + max_row - i];
				L_tri->values[j + max_row - i] = temp[i];

				temp[i] = perm_mat->values[j + i * perm_mat->cols];
				perm_mat->values[j + i * perm_mat->cols] = perm_mat->values[j + max_row - i];
				perm_mat->values[j + max_row - i] = temp[i];
			}
		}

		// Check at A[k, k] value. If the value equals to zero (singular matrix) then exit 
		if (this->values[i + i * this->cols] == 0){
			std::cerr << "Matrix is singular, thus the operator can't be continued" << std::endl;
			return;
		}

		// Reduce Matrix A into triangle matrix
		for (int k = i + 1; k < this->rows; k++){
			s = this->values[i + k * this->cols] / this->values[i + i * this->cols];
			for (int j = 0; j < this->cols - i; j++){
				this->values[j + i + k * this->cols] -= s * this->values[j + i + i * this->cols];
				L_tri->values[i + k * this->cols] = s;
			}
		}

		for (int i = 0; i < cols; i++) 
			L_tri->values[i * cols + i] = 1;

	}
}

template <class T>
void Matrix<T>::forwardSub(T* b, T* y)
{
	double sum = 0;
	// find x vector from lower triangle matrix and vector b.
	for (int i = 0; i < this->rows; i++){
		sum = 0;
		for (int j = 0; j < i; j++)
			sum += this->values[i * this->rows + j] * y[j];
		y[i] = (b[i] - sum) / this->values[i * this->cols + i];
	}
}

template <class T>
void Matrix<T>::backwardSub(T* y, T* x)
{
	double sum = 0;
	// find x vector from upper triangle matrix and vector b.
	for (int i = this->rows - 1; i > -1; i = i - 1){
		sum = 0;
		for (int j = i + 1; j < this->rows; j++)
			sum += this->values[i * this->rows + j] * x[j];
		x[i] = (y[i] - sum) / this->values[i * this->rows + i];
	}
}


