# DNF Regression Solver

This tool generates a DNF expression by regressing the values of input Boolean variables.  Overall, it works by creating DNF separately for true cases and false cases, and then find the common DNF factors.  The run time is exponential due to the parts that the combinations increase factorially as the input variable increases and that it converts fron cnf to dnf.

TO-DO:
In future, it will be upgraded to be able to handle non-Boolean numeric input variables and result variable by discretizing them into segments.

HOW TO RUN:

file_path = '/kaggle/input/tomio5/dnf_regression.txt'

DNF_Regression_solver.solve(file_path)

The input file is a tab-delimited text file where the fields are conditions indicated by 1 or 0, and the last field (or column) indicates the result as sampled below.  Also, a sample file dnf_regression.txt is in the repository.

a	b	c	d	e	f	g	Res

1	1	1	1	1	1	1	1

1	1	1	1	1	1	0	1

1	1	1	1	1	0	1	1
