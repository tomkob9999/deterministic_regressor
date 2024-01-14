# Deterministic Logic Regression Model

This is a deductive ML model based on deterministic approach unlike much of existing models that are stochastic and search for the trend or cleanest divider using the minimum sums of euclidean distances, optimization models using derivatives, etc.  This model instead searches for the logical expression that determins the final true/false decisions.  The existing models cannot seem overcome the complexity of AND/OR relationships of input variables even though ensemble or multi-kernel models like random forests or neural network are able to capture, but they seem to be more of randomized effects than nailing precisely.

One of the advantages, maybe the biggest, is that this derived trained model is 100% explicit and understandable, which is very foreign to regular ML models where the trained models are almost black-box or hardly human understandable.  The derived bool expressions are totally independent from the tool, and they can be analyzed, manually modified or even can be used on any programming languages, or even Excel equations.

This tool generates a boolean expression by regressing the values of input variables.  Overall, it works by creating logic expressions separately for true cases and false cases.

It generates 2 logic expressions by train.
- TRUE DNF: The logic expression is derived for true judgements.  
- FALSE CNF: The logic expression is derived for false judgements.  

It has 4 options for solve.
- UNION: union of TRUE DNF and FALSE CNF
- COMMON: intersection of TRUE DNF and FALSE CNF
- TRUE: TRUE DNF only
- FALSE: FALSE CNF only

The UNION is the default option.  With the confidence threshold properly set, the union option tends to combine the best of both TRUE DNF and FALSE CNF.  The confidence threshold can be set to low or high based on the size of input data during train() (not solve()).  It can be as low as 0.  It is possible to hyper-parameterize by starting low and go higher until the accuracy declines as it tends to find the minimum, but still effective, set of logic clauses, which avoids overfitting and leads to better performance.  Please note train() uses min_match as the threshold already(defaulted to 3), so there is no use in setting confidence_thresh in solve() lower than min_match in train().

Now, it offers optimize_params() function that finds the best parameters using its own logic.

Sample Test:
A sample test has been done for sklearn.datasets.load_breast_cancer dataset (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).  The train data was randomly picked 228 out of 569, and the whole 569 were used as the test data.  The regressor found the logical expression below and the stats.

Derived Expression:
((worst_perimeter <= 100.4)) | ((mean_compactness <= 0.1002 | 0.07262 < worst_concave_points <= 0.1453 | worst_radius <= 14.92) & (mean_concave_points <= 0.04951 | mean_texture <= 17.08 | worst_perimeter <= 100.4) & (mean_concave_points <= 0.04951 | 0.07262 < worst_concave_points <= 0.1453) & (mean_concave_points <= 0.04951 | worst_concavity <= 0.312 | worst_radius <= 14.92) & (mean_concave_points <= 0.04951 | worst_perimeter <= 100.4 | worst_texture <= 21.4) & (mean_concavity <= 0.1065 | 0.07262 < worst_concave_points <= 0.1453) & (worst_concave_points <= 0.07262 | 0.07262 < worst_concave_points <= 0.1453) & (0.07262 < worst_concave_points <= 0.1453 | worst_perimeter <= 100.4 | worst_smoothness <= 0.1089))

Stats:
Precision: 94.05%
Recall: 97.48%
F1 Score: 95.74%
Effectiveness & Efficiency Score: 94.435% (original statistic used in optimize_params())

TO-DO-FUTURE:
- n/a

HOW TO RUN:

file_path = '/kaggle/input/tomio5/dnf_regression.txt'

reg = Deterministic_Regressor()
inp = reg.train(file_path=file_path, error_tolerance=0.03, check_negative=True)

answer = [int(inp[i][-1]) for i in range(1, len(inp), 1)]
inp = [row[:-1] for row in inp]

res = reg.solve(inp, use_expression="false")
print("Predicted")
print(res)
print("Actual")
print(answer)

The input file is a tab-delimited text file where the fields are conditions indicated by 1 or 0, and the last field (or column) indicates the result as sampled below.  Also, a sample file dnf_regression.txt is in the repository.

a	b	c	d	e	f	g	Res

1	1	1	1	1	1	1	1

1	1	1	1	1	1	0	1

1	1	1	1	1	0	1	1

Sample image:

![aa6](https://github.com/tomkob9999/dnf_regression_resolver/assets/96751911/3bc22090-5ed2-46b0-b5bb-a1998b539286)

Here is solve() execution result sample.  Stats are taken by scipy commands (precision_score(), recall_score(), f1_score()).

![aa7](https://github.com/tomkob9999/dnf_regression_solver/assets/96751911/4b45de5d-9288-41b5-b1d6-233e5211af34)

