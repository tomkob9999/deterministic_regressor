# DNF Regression Solver

This tool generates a DNF expression by regressing the values of input Boolean variables.  Overall, it works by creating DNF separately for true cases and false cases, and then find the common DNF factors.  The run time is exponential due to the parts that the combinations increase factorially as the input variable increases and that it converts fron cnf to dnf.

DNF Common should be considered reliable.  DNF TRUE should be considered strong candidates.  DNF FALSE are more of references.

TO-DO:
- enable to handle non-Boolean numeric input variables and result variable by discretizing them into segments
- improve performance when negative insertion is inactive by replacing 0 and 1 for all cells for false DNF calculation

HOW TO RUN:

file_path = '/kaggle/input/tomio5/dnf_regression.txt'

DNF_Regression_solver.solve(file_path)

The input file is a tab-delimited text file where the fields are conditions indicated by 1 or 0, and the last field (or column) indicates the result as sampled below.  Also, a sample file dnf_regression.txt is in the repository.

a	b	c	d	e	f	g	Res

1	1	1	1	1	1	1	1

1	1	1	1	1	1	0	1

1	1	1	1	1	0	1	1

Sample image:

![aa6](https://github.com/tomkob9999/dnf_regression_resolver/assets/96751911/3bc22090-5ed2-46b0-b5bb-a1998b539286)
