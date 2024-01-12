# Deterministic Logic Regression Model

This is a deductive ML model based on deterministic approach unlike much of existing models that are stochastics and search for the trend using the minimum sum of euclidean distances, the optimimum coeeficients by gradient descent or minimizining the entropies to divide segments, etc.  This model instead searches for the logical expression that determins the final true/false decisions.  The existing models cannot overcome the complexity of AND/OR relationships of input variables because the regression models measures the euclidean distances and the deicision tree tries to optimize the information gain for each branch and is not fully deterministic, but this model can search for only the deterministic relations and can find the most inricate relationships.

This tool generates a boolean expression by regressing the values of input Boolean variables.  Overall, it works by creating logic expressions separately for true cases and false cases.

It generates 4 logic expressions by train.  The DNF core is used as default for solve.
- COMMON: Judged true when both TRUE DNF and FALSE CNF match
- TRUE DNF: The logic expression is derived for true judgements.  
- FALSE CNF: The logic expression is derived for false judgements.  
- UNION: Judged true when either TRUE DNF or FALSE CNF matches

Here is the aggressiveness to judge as true.

COMMON < FALSE CNF = TRUE DNF < UNION


TO-DO-FUTURE:
- n/a

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
