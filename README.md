# Linear Solvers Package
acse-5-assignment-morpheus created by GitHub Classroom

C++ class for implementation of several solvers for dense and sparse linear systems.

Team Members: Karim Bacchus, Andika Hakim, Wisit Promrak. 


## Report

Please see report.pdf for our report.

## Compile and Run 

The classes `Matrix.cpp` and `CSRMatrix.cpp` require cblas to be installed on your machine in order to compile. 
On linux and macos, to compile main.cpp, run 

```
gcc main.cpp Matrix.cpp CSRMatrix.cpp -I/your_path/OpenBLAS/include/ -L/your_path/OpenBLAS/lib -lopenblas -o main.out
```

Then execute `./main.out`. 

## Methods in in `Matrix.cpp`

Run the method `solver` as `matrix_A.solver(& b, & x, method, max_iter, tol)` to solve the system Ax = b. 
By default it will use LU decomposition. The optional parameter `method` can specify the following particular solvers: 
- LU Decomposition: `method = "LU"`
- Jacobi iteration: `method = "jacobi"`
- Gauss-Seidel iteration: `method = "gauss"`
- Conjugate Gradient iteration: `method = "cg"`
- GMRES with Jacobi Preconditioner: `method = "gmres"`

The optional parameters `max_iter` and `tol` can be used to set the maximum number of iterations and tolerance required in our iterative methods. This will set the array pointed by `&x` to be the solution to Ax=b. 

## Methods in in `CSRMatrix.cpp`

Run the method `solver` as `matrix_A.solver(& b, & x, method, max_iter, tol)` to solve the system Ax = b. 
By default it will use Gauss Seidel iteration. The optional parameters `method`, `max_iter` and `tol` can specify 
the following:
- Jacobi iteration: `method = "jacobi"`
- Gauss-Seidel iteration: `method = "gauss"`
- Conjugate Gradient iteration: `method = "cg"`
- Cholesky Decomposition : `method = "chol"`

The optional parameters `max_iter` and `tol` can be used to set the maximum number of iterations and tolerance required in our iterative methods. This will set the array pointed by `&x` to be the solution to Ax=b. 

<br>

![Morpheus](https://github.com/karimjbacchus/linear_solvers/blob/master/morpheus.jpg?raw=true)

