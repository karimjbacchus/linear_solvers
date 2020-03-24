#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>
#include "Matrix.h"
#include "Matrix.cpp"
#include "CSRMatrix.h"
#include "CSRMatrix.cpp"
#include <iomanip>
#include <cblas.h>
#include <string>
#include <cstdlib> 

using namespace std;

int main() {

    // Let's load our 10x10 test matrix
    int rows = 10, cols = 10; 

    string input_matrix = "test_A.txt";
    string input_vector = "test_b.txt";

    string csr_values = "csr_values.txt";
    string csr_col_index = "csr_index.txt";
    string csr_row_pos = "csr_row.txt";

    // create matrix A
    unique_ptr<Matrix<double>> mat_A(new Matrix<double>(rows, cols, true));

    int entry = 0, i = 0;
    fstream mymatrix(input_matrix, ios_base::in);
    while (mymatrix >> entry)
    {
        mat_A->values[i] = entry;
        i++;
    }
    mymatrix.close();


    // create vector b
    unique_ptr<double[]> vec_b(new double[rows]);

    for (int i = 0; i < rows; i++)
        vec_b.get()[i] = i+1;

    // create vector x, which will hold our solution
    shared_ptr<double[]> vec_x(new double[rows]);

    
    /***************************************************************/
    // We print the result of each solver and the time taken 
    // to solve Ax = b below
    /***************************************************************/

    cout << "Our Matrix A is: \n";
    mat_A->printMatrix();
    cout << "and our vector b is: ";

    for (int i = 0; i < rows; i++)
        cout << vec_b.get()[i] << " ";

    cout << "\nThe solution for the linear system Ax = b is: \n";

     cout << "\n /**********************************************" << "\n*   START OF DENSE MATRIX SOVLERS \n ";
     cout << "**********************************************/\n\n";


    /***************************************************************/
    // Dense Methods from Matrix.cpp
    /***************************************************************/

    clock_t start, end;

    std::cout << "Using Jacobi method : \n";
    start = clock();

    // Solve Using Jacobi
    mat_A->solver(vec_b.get(), vec_x.get(), "jacobi"); 

    end = clock();

    // Print Jacobi Result
    for (int i = 0; i < rows; i++)
        std::cout << std::setw(10) << vec_x[i];
    std::cout << std::endl << "Time for Jacobi : " << (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1000.0 << " ms \n\n";


    std::cout << "Using Gauss-Seidel method : \n";
    start = clock();

    // Solve Using Gauss-Seidel
    mat_A->solver(vec_b.get(), vec_x.get(), "gauss");

    end = clock();

    // Print Gauss-Seidel Result
    for (int i = 0; i < rows; i++)
        std::cout << std::setw(10) << vec_x[i];
    std::cout << std::endl << "Time for Gauss-Seidel : " << (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1000.0 << " ms \n\n";


    std::cout << "Using GMRES method : \n";
    start = clock();

    // Solve Using GMRES
    mat_A->solver(vec_b.get(), vec_x.get(), "GMRES", 10); 

    end = clock();

    // Print GMRES Result
    for (int i = 0; i < rows; i++)
        std::cout << std::setw(10) << vec_x[i];

    std::cout << std::endl << "Time for GMRES : " << (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1000.0 << " ms \n\n";


    std::cout << "Using CG method : \n";
    start = clock();

    // Solve Using Conjguate Gradient
    mat_A->solver(vec_b.get(), vec_x.get(), "cg"); 
    end = clock();

    // Print Conjguate Gradient Result
    for (int i = 0; i < rows; i++)
        std::cout << std::setw(10) << vec_x[i];
    std::cout << std::endl << "Time for Conjugate Gradident: " << (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1000.0 << " ms \n\n";



    std::cout << "Using LU decomposition with partial pivoting method : \n";

    start = clock();

    // Solve Using LU Decomposition
    mat_A->solver(vec_b.get(), vec_x.get());

    end = clock();

    // Print LU Decomposition Result
    for (int i = 0; i < rows; i++)
        std::cout << std::setw(10) << vec_x[i];

    std::cout << std::endl << "Time for LU Decomposition: " << (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1000.0 << " ms \n\n";



    /***************************************************************/
    // Sparse Methods from CSRMatrix.cpp
    /***************************************************************/

    int nnzs = 80; 
    shared_ptr<CSRMatrix<double>>sparse_mat(new CSRMatrix<double>(rows, cols, nnzs, true));

    // Input CSR Values
    fstream myvalue(csr_values, ios_base::in);
    entry, i = 0;
    while (myvalue >> entry)
    {
         sparse_mat->values[i] = entry;
        i++;
    }
    myvalue.close();

    // add col_index
    fstream mycolumn(csr_col_index, ios_base::in);
    entry, i = 0;
    while (mycolumn >> entry)
    {
         sparse_mat->col_index[i]  = entry;
        i++;
    }
    mycolumn.close();

    // add row_positions
    fstream myrow(csr_row_pos, ios_base::in);
    entry, i = 0;
    while (myrow >> entry)
    {
         sparse_mat->row_position[i] = entry;
        i++;
    }
    myrow.close();


cout << "\n /**********************************************" << "\n*   BEGIN OF SPARSE MATRIX SOLVER\n ";
cout << "**********************************************/\n";

    std::cout << "Using Sparse Jacobi method : \n";
    start = clock();

    // Solve Using Sparse Jacobi
    sparse_mat->solver(vec_b.get(), vec_x.get(), "jacobi"); 
    end = clock();
    
    // Print Sparse Jacobi Result
    for (int i = 0; i < rows; i++)
        std::cout << std::setw(10) << vec_x[i];

    std::cout << std::endl << "Time : " << (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1000.0 << " ms\n\n";


    std::cout << "Using Sparse Gauss-Seidel method : \n";
    start = clock();

    // Solve Using Sparse Gauss-Seidel
    sparse_mat->solver(vec_b.get(), vec_x.get(), "gs"); 
    end = clock();
    
    // Print Sparse Gauss-Seidel Result
    for (int i = 0; i < rows; i++)
        std::cout << std::setw(10) << vec_x[i];

    std::cout << std::endl << "Time : " << (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1000.0 << " ms\n\n";



    std::cout << "Using Sparse Conjguate Gradient method : \n";
    start = clock();

    // Solve Using Sparse Conjguate Gradient
    sparse_mat->solver(vec_b.get(), vec_x.get(), "cg"); 
    end = clock();
    
    // Print Sparse Conjguate Gradient Result
    for (int i = 0; i < rows; i++)
        std::cout << std::setw(10) << vec_x[i];

    std::cout << std::endl << "Time : " << (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1000.0 << " ms\n\n";





    // For Sparse Cholesky, we assume the CSR matrix only stores the lower triangular
    // entries of the matrix, since it is only valid for symmetric definite positive matrices. 
    // So we load in another CSR Matrix that is the same as the previous one except
    // it only contains the lower triangular entries to test for our Sparse Cholesky solver. 

    string sym_csr_values = "csr2_values.txt";
    string sym_csr_col_index = "csr2_index.txt";
    string sym_csr_row_pos = "csr2_row.txt";

    int sym_nnzs = 45; 
    shared_ptr<CSRMatrix<double>>sparse_mat2(new CSRMatrix<double>(rows, cols, sym_nnzs, true));

    // Input CSR Values
    fstream myvalue2(sym_csr_values, ios_base::in);
    entry, i = 0;
    while (myvalue2 >> entry)
    {
         sparse_mat2->values[i] = entry;
        i++;
    }
    myvalue2.close();

    // add col_index
    fstream mycolumn2(sym_csr_col_index, ios_base::in);
    entry, i = 0;
    while (mycolumn2 >> entry)
    {
         sparse_mat2->col_index[i]  = entry;
        i++;
    }
    mycolumn2.close();

    // add row_positions
    fstream myrow2(sym_csr_row_pos, ios_base::in);
    entry, i = 0;
    while (myrow2 >> entry)
    {
         sparse_mat2->row_position[i] = entry;
        i++;
    }
    myrow2.close();

    std::cout << "Using Sparse Cholesky method : \n";
    start = clock();

    // Solve Using Sparse Cholesky
    sparse_mat2->solver(vec_b.get(), vec_x.get(), "cholesky"); 
    end = clock();
    
    // Print Sparse Conjguate Gradient Result
    for (int i = 0; i < rows; i++)
        std::cout << std::setw(10) << vec_x[i];

    std::cout << std::endl << "Time : " << (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1000.0 << " ms\n\n";

}