# DNF Regression Solver

This script does the inversion of the above.  The above one goes DNF -> datasets, and this one goes datasets -> DNF.  It finds the bool expression from the input data like Linear Regression finds coefficients of covectors.  It does not use either linear regression or decision tree logic, though.  Instead, it looks for DNF by matching the true and false identitity records, which is the essence used in DNF Test Creator as well.  The runtime has degraded from linear to non-linear since 1.4 upgrade to accomodate realistic data that do not contain Cartesian product of variable values.


HOW TO RUN:

file_path = '/kaggle/input/tomio5/dnf_regression.txt'

DNF_Regression_solver.solve(file_path)

The input file is a tab-delimited text file where the fields are conditions indicated by 1 or 0, and the last field (or column) indicates the result as sampled below.  Also, a sample file dnf_regression.txt is in the repository.

a	b	c	d	e	f	g	Res

1	1	1	1	1	1	1	1

1	1	1	1	1	1	0	1

1	1	1	1	1	0	1	1
