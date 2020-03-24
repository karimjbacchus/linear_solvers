/* 
 * File: Matrix.h
 * --------------
 * This interface exports the Matrix class 
 */

#pragma once
#include <vector>
#include <memory>
#include <string>

template <class T> // declare template T for class Matrix
class Matrix
{
public:

    // constructor where we want to preallocate memory, own its own memory
    Matrix(int rows, int cols, bool preallocate);
    // constructor where we have already preallocated memory outside, dont own memory
    Matrix(int rows, int cols, T* values_ptr);

    // destructor
    virtual ~Matrix(); // virtual = subclass/inherit can be overwrite

    // print the values of matrix
    void printValues();
    virtual void printMatrix(); // virtual since will be different for sparse matrix subclass


    /*
    * Method: matMatMult
    * Useage: mat_A.matMatMult(mat_B, mat_C); 
    * ----------------------------------------
    * Sets matrix mat_C to be the matrix product mat_A*mat_B
    */
    virtual void matMatMult(Matrix<T>& mat_right, Matrix<T>& output); // polymorphism; will be different in CSRMatrix

    /*
    * Operator: +
    * Useage: Matrix mat_sum = mat_A + mat_B
    * ---------------------------------------
    * Overloads the + operator so that it can also add matrices
    */
    Matrix operator+(Matrix<T>& mat_right);

    /*
    * Method: matVectMult
    * Useage: mat_A.matMatMult(&b , &c);
    * ----------------------------------------
    * Sets vector c to be the matrix-vector product mat_A*b
    */           
    virtual void matVecMult(T* vect_right, T* output); // polymorphism; will be different in CSRMatrix


    /*
    * Method: solver
    * Useage: mat_A.solver(x, b)
    * ---------------------------
    * Sets the matrix x to be the solution to the linear system Ax = b
    * Takes in method as optional parameter - default method is LU Decomposition, 
    * can also call Jacobi, Gauss, GMRES and Conjugate Gradient methods.
    */
    void solver(const T* vec_b, T* vec_x, std::string method = "LU", int max_iter = 10000, double tol = 1e-5);


    /*
    * Method: LU decomposition solver
    * Useage: mat_A.luSolve(x, b)
    * ---------------------------
    * Sets the matrix x to be the solution to the linear system Ax = b
    * using LU Decomposition with partial pivoting. It calls 
    * lupDecompose, forwardSub and then backwardSub
    */  
    void luSolve(const T* b_vect, T* x_vect);


    /*
    * Method: LU decomposition solver
    * Useage: mat_A.luDecompose(&mat_L, &mat_P)
    * ---------------------------
    * Sets L, P, such that A = LUP
    */   
    void lupDecompose(Matrix<T>* L_tri, Matrix<T>* perm_mat);
    void forwardSub(T* b, T* y);
    void backwardSub(T* y, T* x);


    /*
    * Method: Iterative Sparse Solvers
    * Useage: mat_A.jacobi(&b, &x, 10000, 1e-5)
    * -------------------------------------------
    * Sets the matrix x to be the solution to the linear system Ax = b 
    * using iterative solvers below
    */
    void cg(const T* b, T* x, int max_iter = 1000, double tol = 1e-5);
    void jacobi(const T* b, T* x, int max_iter = 5000, double tol = 1e-5);
    void gauss(const T* b, T* x, int max_iter = 50000, double tol = 1e-5);
    void gmres(const T* b, T* x, int max_iter = 10000, double tol = 1e-5);


    /*
    * Method: leastSquares
    * Useage: A.leastSquares(x, b)
    * -------------------------------
    * Sets x to to minimize |Ax - b|, using QR factorisation of A during gmres
    * routine, taking in prior Givens rotations G.
    */
    double leastSquares(T* x, T* b, Matrix<T> &G, int n); 


    /// public variable
    /* explicitly using c++11 nullptr*/
    std::shared_ptr<T[]> values{nullptr};
    int rows = -1;
    int cols = -1;


protected:
    int size_of_values = -1;
    bool preallocated = false;

private:

};