### Name: Deterministic_Regressor
# Author: tomio kobayashi
# Version: 3.3.0
# Date: 2024/02/18

import itertools
from sympy.logic import boolalg
import numpy as np
import sklearn.datasets
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sympy import simplify
import copy
from collections import Counter
import time
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score, classification_report
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning


class Deterministic_Regressor:
# This version has no good matches
# Instead, all true matches are first added, and some are removed when 
# false unmatch exists and there is no corresponding other rule
    def __init__(self, supress_sklearn_warn=True):
        self.expression_true = ""
        self.expression_false = ""
        self.true_confidence = {}
        self.false_confidence = {}
        self.all_confidence = {}
        
        self.tokens = []
        self.dic_segments = {}
        
        self.last_solve_expression = ""
        
        self.expression_opt = ""
        self.by_two = -1
        self.opt_f1 = 0.001
        
        self.children = []
        
        self.whole_rows = []
        self.test_rows = []
        self.train_rows = []

        self.whole_rows_org = []
        self.test_rows_org = []
        self.train_rows_org = []
        
        self.classDic = {}
        self.classDicRev = {}
        self.item_counts = {}
        
        self.combo_list = []
        self.predictors = []
        self.target_cols = []
        self.gmm = None

        if supress_sklearn_warn:
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    def IsNonBinaryNumeric(n):
        try:
            return any([nn != 1 and nn != 0 and nn < float('inf') for nn in n])
        except Exception as e:
            return False

    def findClusters(X_in, max_clusters=10):

        X = np.array([np.array(x) for x in X_in])
        target_cols = [i for i, xx in enumerate(X[0]) if Deterministic_Regressor.IsNonBinaryNumeric([row[i] for row in X])]

        n_components = np.arange(1, max_clusters)
        models = [GaussianMixture(n, covariance_type='full').fit(X) for n in n_components]
        aic = [model.aic(X) for model in models]
        bic = [model.bic(X) for model in models]

        num_clusters = 1
#         for j in range(len(models)-1):
#             if j == 0 and (aic[0] < aic[1] or bic[0] < bic[1]):
#                 break
#             if j > 0 and (aic[j] < aic[j+1] or bic[j] < bic[j+1]):
#                 num_clusters = j+1
#                 break
        min_aic = float("inf")
        min_bic = float("inf")
        for j in range(len(models)):
            if aic[j] > min_aic and bic[j] > min_bic:
                num_clusters = j
                break
            if aic[j] < min_aic:
                min_aic = aic[j] 
            if bic[j] < min_bic:
                min_bic = bic[j] 
        X_clust = []
        y_clust = []
        if num_clusters > 1:
            gmm = GaussianMixture(n_components=num_clusters)
            gmm.fit(X)
            cluster_labels = gmm.predict(X)
            return cluster_labels, gmm
        else:
            return None, None

    def calculate_correlation_matrix(data):
        """Calculate the correlation matrix for the given data."""
        standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        n = data.shape[0]
        corr_matrix = np.dot(standardized_data.T, standardized_data) / n
        return corr_matrix

    

    def give_correlated_columns_to_y(data, y, threshold=0.7):
        """Remove columns with strong correlations."""
#         print("aaa", [row + [y[i]] for i, row in enumerate(data)])
        corr_matrix = Deterministic_Regressor.calculate_correlation_matrix(np.array([row + [y[i]] for i, row in enumerate(data)]))
        related_cols = set()

        ind_y = len(corr_matrix)-1
        for j in range(corr_matrix.shape[1]-1):
            if abs(corr_matrix[ind_y, j]) > threshold:
                related_cols.add(j)

        return sorted(related_cols)

    def remove_highly_correlated_columns(data, threshold=0.8):
        """Remove columns with strong correlations."""
        corr_matrix = Deterministic_Regressor.calculate_correlation_matrix(data)
        columns_to_remove = set()

        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > threshold:
                    # Mark the column with the higher index for removal
                    columns_to_remove.add(j)

        print("columns_to_remove", columns_to_remove)
        # Create a new array without the highly correlated columns
        reduced_data = np.delete(data, list(columns_to_remove), axis=1)
        return reduced_data


    def give_highly_correlated_columns(data, target_cols, threshold=0.95):
        """Remove columns with strong correlations."""
        corr_matrix = Deterministic_Regressor.calculate_correlation_matrix(data)
        columns_to_remove = {}

        for i in range(corr_matrix.shape[0]):
            if i in target_cols:
                continue
            for j in range(i+1, corr_matrix.shape[1]):
                if j in target_cols:
                    continue
                if abs(corr_matrix[i, j]) > threshold:
                    # Mark the column with the higher index for removal
                    columns_to_remove[j] = i
        return columns_to_remove


    def covariance_matrix_multi(columns, matrix, nvar):

        dat = np.copy(matrix)
        dat_T = dat.T
        newnewdata_T = list(copy.deepcopy(dat_T))
        lastdata_T = list(copy.deepcopy(dat_T))
        newdata_T = list(copy.deepcopy(dat_T))
        num_loop = nvar-1 if nvar < len(columns) else len(columns)-1
        tup_cols = [(c,) for c in columns]
        new_cols = [(c,) for c in columns]
        start_time = time.time()
        cnt = 0
        for _ in range(num_loop):
            if len(dat_T) * len(newnewdata_T) > 100000:
                print("Loop stops at", _+1, "as too many")
                break
            lastdata_T = copy.copy(newnewdata_T)
            newnewdata_T = []
            for i in range(len(dat_T)):
                start_time = time.time()
                for j in range(len(lastdata_T)):
                    cnt += 1
                    if tup_cols[i][0] in set(list(new_cols[j])) or tuple(sorted(list(new_cols[j] + tup_cols[i]))) in new_cols:
                        continue
                    newnewdata_T.append(dat_T[i]*newdata_T[j])
                    new_cols.append(tuple(sorted(list(new_cols[j] + tup_cols[i]))))
            for n in newnewdata_T:
                newdata_T.append(n)
        return np.array(newdata_T), new_cols

    def derive_dnf_stochastic(columns, matrix, nvar, answer, cut_off=0.00):
        numrecs = len(matrix)
        new_dat, new_cols = Deterministic_Regressor.covariance_matrix_multi(columns, matrix, nvar)
        sum_bef = np.array([sum(n) for n in new_dat])
        sum_aft = np.dot(new_dat, answer)
        res = np.array([sum_aft[i] / sum_bef[i] if sum_bef[i]/numrecs >= cut_off else 0.00 for i in range(len(sum_aft))])
        dic_head = {columns[i]:i for i in range(len(columns))}
        dic_ncols = {n: tuple([dic_head[nn] for nn in n]) for n in new_cols}
        return res, new_cols, dic_ncols, sum_bef

    def remove_supersets(sets_list):
        result = []
        for s in sets_list:
            if not any(s != existing_set and s.issuperset(existing_set) for existing_set in sets_list):
                result.append(s)
        return result


    def cnf_to_dnf_str(str):
        ss = str.split("&")
        ss = [a.strip()[1:-1] if a.strip()[0] == "(" else a for a in ss]
        cnf = [[b.strip() for b in sa.strip().split("|")] for sa in ss]

        dnf = []
        for clause in cnf:
            dnf_clause = []
            for literal in clause:
                dnf_clause.append(literal)
            dnf.append(dnf_clause)
        dnfl = [list(x) for x in itertools.product(*dnf)]
        dnfl = [set(d) for d in dnfl]
        filtered_sets = Deterministic_Regressor.remove_supersets(dnfl)
        filtered_lists = [sorted(list(f)) for f in filtered_sets]
        filtered_lists = set([" & ".join(f) for f in filtered_lists])
        str = "(" + ") | (".join(sorted(filtered_lists)) + ")"
        return str
    
    
    def simplify_dnf(s, use_cnf=False, withSort=False):
        tok1 = "|"
        tok2 = "&"
        if use_cnf:
            tok1 = "&"
            tok2 = "|"
        if s.strip() == "":
            return ""
        ss = s.split(tok1)
        ss = [s.strip() for s in ss]
        ss = [s[1:-1] if s[0] == "(" else s for s in ss]
        ss = [s.split(tok2) for s in ss]
        ss = [[sss.strip() for sss in s] for s in ss]
        ss = [set(s) for s in ss]

        filtered_sets = Deterministic_Regressor.remove_supersets(ss)
        filtered_lists = [sorted(list(f)) for f in filtered_sets]
        filtered_lists = [(" " + tok2 + " ").join(f) for f in sorted(filtered_lists)] if withSort else [(" " + tok2 + " ").join(f) for f in filtered_lists]
        str = "(" + (") " + tok1 + " (").join(filtered_lists) + ")"
        return str

    def try_convert_to_numeric(text):
        if isinstance(text, str):
            if "." in text:
                try:
                    return float(text)
                except ValueError:
                    pass
            else:
                try:
                    return int(text)
                except ValueError:
                    pass
        return text

    def convTuple2bin(t, width):
        i = 1
        ret = 0
        for x in t:
            ii = i << width - x - 1
            ret = ret | ii
        return ret
    
    def myeval(__ccc___, __tokens___, ______s____):
        for __jjjj___ in range(len(__tokens___)):
            exec(__tokens___[__jjjj___] + " = " + str(__ccc___[__jjjj___]))
        return eval(______s____)
            
    def solve(self, inp_p, confidence_thresh=0, power_level=None, useUnion=False):
        
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp_p[i][j]) for j in range(len(inp_p[i]))] for i in range(len(inp_p))]
        
        
        max_power = 64
        
        all_confidence_thresh = 0
        if power_level is not None:
            if power_level > max_power or power_level < 0:
                print("power_level must be between 0 and 32")
                return

            if len(self.all_confidence) == 0:
                    all_confidence_thresh = 0  
            else:
                max_freq = max([v for k, v in self.all_confidence.items()])
                print("max_freq", max_freq)
                min_freq = min([v for k, v in self.all_confidence.items()])
                this_power = int(power_level/max_power * (max_freq-min_freq) + 0.9999999999)

                if max_freq - this_power < 0:
                    all_confidence_thresh = 0
                else:
                    all_confidence_thresh = max_freq - this_power

        else:
            all_confidence_thresh = confidence_thresh
        print("Confidence Thresh:", all_confidence_thresh)
        print("Input Records:", len(inp)-1)
        
        numvars = len(inp[0])

        tokens = inp[0]
        inp_list = [row for row in inp[1:]]
        
        res = list(range(len(inp_list)))
        expr = ""
        
        true_exp = self.expression_true
        false_exp = self.expression_false
        active_true_clauses = 0
        active_false_clauses = 0
        if all_confidence_thresh > 0:
            true_list = []
            for s in true_exp.split("|"):
                s = s.strip()
                if s in self.true_confidence:
                    if self.true_confidence[s] >= all_confidence_thresh:
                        true_list.append(s)
                        active_true_clauses += 1
            true_exp = " | ".join(true_list)
            
            false_list = []
            for s in false_exp.split("|"):
                s = s.strip()
                if s in self.false_confidence:
                    if self.false_confidence[s] >= all_confidence_thresh:
                        false_list.append(s)
                        active_false_clauses += 1
            false_exp = " | ".join(false_list)
        else:
            active_true_clauses = len(true_exp.split("|"))
            active_false_clauses = len(false_exp.split("|"))

        connector = " | " if useUnion else " & "

        if true_exp != "":
            expr = "(" + true_exp + ")"
        if true_exp != "" and false_exp != "":
            expr = expr + connector
        if false_exp != "":
            expr = expr + "not (" + false_exp + ")"
            
        print(str(active_true_clauses) + " true clauses activated")
        print(str(active_false_clauses) + " false clauses activated")
            
        expr = expr.replace(" & ", " and ").replace(" | ", " or ")
        print("Solver Expression:")
        
        print(self.replaceSegName(expr))
        
        self.last_solve_expression = expr

        if len(expr) == 0:
            print("No expression found")
            return []
        
        for i in range(len(inp_list)):
            res[i] = Deterministic_Regressor.myeval(inp_list[i], tokens, expr)
            
        if res is None:
            return []
        
        return [1 if r or r == 1 else 0 for r in res]
    
        
    def solve_direct(self, inp_p, expression):
        
        expression = expression.replace(" | ", " or ").replace(" & ", " and ")
        self.last_solve_expression = expression
           
        inp = copy.deepcopy(inp_p)
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]

        numvars = len(inp[0])
        
        tokens = inp[0]
        inp_list = [row for row in inp[1:]]
        
        res = list(range(len(inp_list)))
        
        for i in range(len(inp_list)):
            res[i] = Deterministic_Regressor.myeval(inp_list[i], tokens, expression)
        return res
    
            
    def solve_with_opt(self, inp_p):
        if self.expression_opt == "":
            return [0] * (len(inp_p)-1)
        else:
            return self.solve_direct(inp_p, self.expression_opt)
    
    def replaceSegName(self, str):
        s = str
        for t in self.tokens:
            if t in s and t in self.dic_segments:
                s = s.replace(t, self.dic_segments[t])
        return s
            
            
    def generate_segment_ranks(self, df, num_segments, name, silent=False):

        df[name + '_rank'] = pd.cut(df[name], bins=num_segments, labels=False)
        df[name + '_label'] = pd.cut(df[name], bins=num_segments, labels=[f'{name} {i+1}' for i in range(num_segments)])
        min_max_per_group = df.groupby(name + '_rank')[name].agg(['max'])
        
        max_list = min_max_per_group.values.tolist()
        prev_max_str = ""
        ranks = sorted(df[name + '_rank'].unique().tolist())
        for i in range(len(max_list)):
            if i == 0:
                self.dic_segments[name + "_" + str(ranks[i])] = name + " <= " + str(max_list[i][0])
                prev_max_str = str(max_list[i][0])
            elif i == len(max_list) - 1:
                self.dic_segments[name + "_" + str(ranks[i])] = prev_max_str + " < " + name
            else:
                self.dic_segments[name + "_" + str(ranks[i])] = prev_max_str + " < " + name + " <= " + str(max_list[i][0])
                prev_max_str = str(max_list[i][0])
            
            
        if not silent:
            print("")
            print(min_max_per_group)
        
        return df

    def discretize_data(self, data_list, by_two=2, silent=False):

        result_header = data_list[0][-1]
        result_values = [d[-1] for d in data_list[1:]]
        
        headers = data_list[0][:-1]
        values = [d[:-1] for d in data_list[1:]]
        data = pd.DataFrame(values, columns=headers) 

        
        cols = [c for c in data.columns]
        for c in cols:
            countNonBool = len(data[c]) - (data[c] == 0).sum() - (data[c] == 1).sum()
            if countNonBool > 0 and pd.api.types.is_numeric_dtype(data[c]):
                result_df = self.generate_segment_ranks(data, by_two*2, c, silent=silent)
                one_hot_df = pd.get_dummies(result_df[c + '_rank'], prefix=c)
                one_hot_df = one_hot_df.astype(int)
                data = pd.concat([result_df, one_hot_df], axis=1)
        cols = [c for c in data.columns]
        new_cols = []
        for c in cols:
            countNonBool = len(data[c]) - (data[c] == 0).sum() - (data[c] == 1).sum()
            if countNonBool == 0 and pd.api.types.is_numeric_dtype(data[c]):
                new_cols.append(c)
        
        data = data.filter(items=new_cols)

        data_list = [data.columns.tolist()] + data.values.tolist()

        
        data_list[0].append(result_header)
        for i in range(len(result_values)):
            data_list[i+1].append(result_values[i])
        
        return data_list

    def reduce_rows_except_first(matrix, percentage):
        if not (0 <= percentage <= 100):
            raise ValueError("Percentage must be between 0 and 100")

        # Ensure the first row is always included
        num_rows_to_keep = max(1, int(len(matrix) * (1 - percentage / 100)))

        # Sample remaining rows
        sampled_rows = [matrix[0]] + random.sample(matrix[1:], num_rows_to_keep - 1)

#         return sampled_rows
        return copy.deepcopy(sampled_rows)


    
    def clean_and_discretize(self, inp, by_two):
        inp = copy.deepcopy(inp)
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]
        matrix = self.discretize_data(inp, by_two)
        head = matrix[0]
        return [head] + [[int(mm) for mm in m] for m in matrix[1:]]
        
    def train(self, file_path=None, data_list=None, max_dnf_len=4, error_tolerance=0.00, min_match=0.00, use_approx_dnf=False, redundant_thresh=1.00, 
              useExpanded=True, use_stochastic=False, stochastic_min_rating=0.97):

        # file_path: input file in tab-delimited text
        # data_list: list matrix data with header in the first row and the result in the last col
        # max_dnf_len: max length of AND clauses in output DNF.  
        #       Increasing this one is heavey especially if check_negative is True

        # Example usage:
        # file_path = '/kaggle/input/dnf-regression/dnf_regression.txt'
        # file_path = '/kaggle/input/tomio5/dnf_regression.txt'
        
        inp = None
        if file_path is not None:
            with open(file_path, 'r') as f:
                inp = [line.strip().split('\t') for line in f]
        else:
            inp = data_list
        
        print("Train Records:", len(inp)-1)
    
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]

        for r in inp[1:]:
            for c in r:
                if c != 1 and c != 0:
                    print("The data contents needs to be 1 or 0", c)
                    return []
        
        imp_before_row_reduction = copy.deepcopy(inp)
        
        numvars = len(inp[1])-1

        print("Columns:")
        print(inp[0])
        self.tokens = copy.deepcopy(inp[0])
        print("")
        
        print("Uni-valued Columns:")
        numrows = len(inp)-1
        redundant_cols = set()
        for i in range(len(inp[0])-1):
            if i in redundant_cols:
                continue
            vals = [row[i] for row in inp[1:]]
            cnts = Counter(vals)
#             print("vals", vals)
#             print("cnts", cnts)
            if len(cnts) == 1:
                redundant_cols.add(i)
                print(self.tokens[i])
        print("")
        
        print("Redundant Pairs:")
        for i in range(len(inp[0])-1):
            for j in range(i+1, len(inp[0])-1):
                if j in redundant_cols:
                    break
                sames = len([1 for k in range(1, len(inp), 1) if inp[k][i] == inp[k][j]])
                if sames/numrows >= redundant_thresh:
                    print(self.tokens[j], "->",self.tokens[i])
                    redundant_cols.add(j)
        print("")
        
        
        
        
        if max_dnf_len > numvars - 1:
            max_dnf_len = numvars - 1

                
        MAX_REPS = 1500000
        
        print("Deriving expressions...")
        dnf_perf = list()
        raw_perf = list()
        raw_perf2 = list()
        dnf_perf_n = list()
        raw_perf_n = list()
        raw_perf2_n = list()

        patterns = [(1, 1), (0, 1), (1, 0), (0, 0)] if useExpanded else [(1, 1), (0, 0)]
        for pattern in patterns:
            print("Starting", pattern)
            dat_t = []
            if pattern[0] == 1:
                dat_t = [row[:-1] for row in inp[1:]]
            else:
                dat_t = [[0 if m == 1 else 1 for m in row[:-1]] for row in inp[1:]]
            res_t = []
            if pattern[1] == 1:
                res_t = [row[-1] for row in inp[1:]]
            else:
                res_t = [0 if row[-1] == 1 else 1 for row in inp[1:]]

            if use_stochastic:
#                 replace_0 = 0.0001
#                 replace_1 = 0.9999
                replace_0 = 0.00001
#                 replace_1 = 0.99999
                replace_1 = 1.
                for i in range(len(dat_t)):
                    for j in range(len(dat_t[0])):
                        dat_t[i][j] = replace_1 if dat_t[i][j] == 1 else replace_0
                        
                for i in range(len(res_t)):
                    res_t[i] = replace_1 if res_t[i] == 1 else replace_0

                cov, ncols, dic_ncols, sum_bef = Deterministic_Regressor.derive_dnf_stochastic(inp[0][:-1], dat_t, max_dnf_len, res_t, min_match)
                new_covs = [dic_ncols[ncols[i]] for i in range(len(cov)) if cov[i] > stochastic_min_rating]
                new_covs = Deterministic_Regressor.remove_supersets([set(list(nn)) for nn in new_covs])
                new_covs = [tuple(nn) for nn in new_covs]
                
                for i in range(len(new_covs)):
                    if pattern[1] == 1:
                        if pattern[0] == 1:
                            key = "(" + " & ".join(sorted(list(set([inp[0][ii] for ii in new_covs[i]])))) + ")"
#                             print("key", key)
                            dnf_perf.append(sorted(list(set([inp[0][ii] for ii in new_covs[i]]))))
                        else:
                            key = "(not (" + ") & not (".join(sorted(list(set([inp[0][ii] for ii in new_covs[i]])))) + "))" 
#                             print("key", key)
                            dnf_perf.append(sorted(list(set(["not (" + inp[0][ii] + ")" for ii in new_covs[i]]))))
                        self.true_confidence[key] = sum_bef[i]

                    else:
                        raw_perf_n.append([ii for ii in new_covs[i]])
#                         raw_perf2_n.append(b)  
                        if pattern[0] == 1:
                            key = "(" + " & ".join(sorted(list(set([inp[0][ii] for ii in new_covs[i]])))) + ")"
#                             print("key", key)
                            dnf_perf_n.append(sorted(list(set([inp[0][ii] for ii in new_covs[i]]))))
                        else:
                            key = "(not (" + ") & not (".join(sorted(list(set([inp[0][ii] for ii in new_covs[i]])))) + "))" 
#                             print("key", key)
                            dnf_perf_n.append(sorted(list(set(["not (" + inp[0][ii] + ")" for ii in new_covs[i]]))))
                        self.false_confidence[key] = sum_bef[i]
                            
            else:
                                         
                inp_t = [copy.deepcopy(inp[0])] + [dat_t[i] + [res_t[i]] for i in range(len(dat_t))]
                dic_t = dict()
                for i in range(1, len(inp_t), 1):
                    s = ""
                    for j in range(len(inp_t[i]) - 1):
                        s += str(int(inp_t[i][j]))
                    truefalse = inp_t[i][len(inp_t[i]) - 1]
                    dic_t[int(s, 2)] = truefalse

                
                for s in range(max_dnf_len):
                    len_dnf = s + 1

                    l = [ii for ii in range(numvars)]
                    p_list = list(itertools.combinations(l, len_dnf))

                    p_list = [p for p in p_list if not any([pp in redundant_cols for pp in p])]

                    print(str(len_dnf) + " variable patterns")
                    if len(p_list) > MAX_REPS:
                        print("Skipping because " + str(len(p_list)) + " combinations is too many")
                        break
                    true_test_pass = True
                    for i in range(len(p_list)):
                        match_and_break = False
                        b = Deterministic_Regressor.convTuple2bin(p_list[i], numvars)
                        if pattern[1] == 1:
                            for p in raw_perf2:
                                if p == b & p:
                                    match_and_break = True
                                    break
                        else:
                            for p in raw_perf2_n:
                                if p == b & p:
                                    match_and_break = True
                                    break
                        if match_and_break:
                            continue
                        r = [v for k,v in dic_t.items() if k & b == b]
                        if len(r) == 0:
                            continue
                        cnt_all = len([f for f in r])
                        cnt_unmatch = len([f for f in r if f == 0])
                        if cnt_unmatch/cnt_all > error_tolerance:
                            continue
                        if (cnt_all - cnt_unmatch)/numrows < min_match:
                            continue

                        if pattern[1] == 1:
                            raw_perf.append([ii for ii in p_list[i]])
                            raw_perf2.append(b)

                            if pattern[0] == 1:
                                key = "(" + " & ".join(sorted(list(set([inp[0][ii] for ii in p_list[i]])))) + ")"
                                dnf_perf.append(sorted(list(set([inp[0][ii] for ii in p_list[i]]))))
                            else:
                                key = "(not (" + ") & not (".join(sorted(list(set([inp[0][ii] for ii in p_list[i]])))) + "))" 
                                dnf_perf.append(sorted(list(set(["not (" + inp[0][ii] + ")" for ii in p_list[i]]))))
                            self.true_confidence[key] = cnt_all - cnt_unmatch

                        else:
                            raw_perf_n.append([ii for ii in p_list[i]])
                            raw_perf2_n.append(b)  
                            if pattern[0] == 1:
                                key = "(" + " & ".join(sorted(list(set([inp[0][ii] for ii in p_list[i]])))) + ")"
                                dnf_perf_n.append(sorted(list(set([inp[0][ii] for ii in p_list[i]]))))
                            else:
                                key = "(not (" + ") & not (".join(sorted(list(set([inp[0][ii] for ii in p_list[i]])))) + "))" 
                                dnf_perf_n.append(sorted(list(set(["not (" + inp[0][ii] + ")" for ii in p_list[i]]))))
                            self.false_confidence[key] = cnt_all - cnt_unmatch

        
        self.all_confidence = copy.deepcopy(self.true_confidence)
        self.all_confidence.update(self.false_confidence)

#         print("size of false dnf " + str(len(dnf_perf_n)))
        
        set_dnf_true = set(["(" + s + ")" for s in [" & ".join(a) for a in dnf_perf]])
        set_cnf_false = set(["(" + s + ")" for s in [" & ".join(a) for a in dnf_perf_n]])

        self.expression_true = " | ".join(set_dnf_true)
        self.expression_true = Deterministic_Regressor.simplify_dnf(self.expression_true)
        self.expression_false = " | ".join(set_cnf_false)
        self.expression_false = Deterministic_Regressor.simplify_dnf(self.expression_false)
        

            
        print("")
        print("TRUE DNF - " + str(len(set_dnf_true)))
        print("--------------------------------")

        if len(set_dnf_true) > 0:
            print(self.replaceSegName(self.expression_true))
            
        print("")
        print("FALSE DNF - " + str(len(set_cnf_false)))
        print("--------------------------------")
        if len(set_cnf_false) > 0:
            print(self.replaceSegName(self.expression_false))
            
        perm_vars = list(set([xx for x in dnf_perf for xx in x] + [xx for x in dnf_perf_n for xx in x]))
        
        not_picked = [self.replaceSegName(inp[0][ii]) for ii in range(len(inp[0])-1) if inp[0][ii] not in perm_vars]

        print("")
        print("Unsolved variables - " + str(len(not_picked)) + "/" + str(len(inp[0])-1))
        print("--------------------------------")
        print(not_picked)
        print("")
        
        return imp_before_row_reduction


    def optimize_params(self, test_data, answer, elements_count_penalty=1.0, useUnion=False):
        
        inp = test_data
        best_ee_sofar = -1
        ct_now = 0

        MAX_POWER_LEVEL = 64
        jump = int(MAX_POWER_LEVEL/2)
        ct_opt = 0
        expr_opt = ""
        opt_precision_sofar = 0
        opt_recall_sofar = 0
        opt_f1_sofar = 0
        opt_match_rate_sofar = 0
        
        doexit = False
        for i in range(100):
            print("")
            print("")
            print("##### Power Level", ct_now, "######")
        
            best_ee = 0
            win_expr = ""
            opt_precision = 0
            opt_recall = 0
            opt_f1 = 0
            opt_match_rate = 0
            res = self.solve(inp, power_level=ct_now, useUnion=useUnion)

            if len(res) == 0:
                print("#################################")
                print("")
                print("SORRY NO SOLUTION FOUND")
                print(str(sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])) + "/" + str(len(res)), " records matched")
                print("")
                print("#################################")
                print("")
                return None, None
            else:

                win_expr = self.last_solve_expression
                num_match = sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])
                print(str(num_match) + "/" + str(len(res)), " records matched " + f" ({num_match/len(res)*100:.2f}%)")        
                precision = precision_score(answer, res)
                recall = recall_score(answer, res)
                f1 = f1_score(answer, res)
                print(f"Precision: {precision * 100:.2f}%")
                print(f"Recall: {recall * 100:.2f}%")
                print(f"F1 Score: {f1 * 100:.2f}%")
                ee = (f1 +min(precision,recall))/2-(len(self.last_solve_expression.split("&"))+len(self.last_solve_expression.split("|")))/3000*elements_count_penalty
                ee = 0 if ee < 0 else ee
                print(f"Effectiveness & Efficiency Score: {ee * 100:.3f}%")
                best_ee = ee
                opt_precision = precision
                opt_recall = recall
                opt_f1 = f1
                opt_match_rate = num_match/len(res)
                if best_ee_sofar < best_ee:
                    ct_opt = ct_now
                    best_ee_sofar = best_ee
                    ct_now = ct_now + jump
                    expr_opt = win_expr
                    opt_precision_sofar = opt_precision
                    opt_recall_sofar = opt_recall
                    opt_f1_sofar = opt_f1
                    opt_match_rate_sofar = opt_match_rate
                    if ct_now > MAX_POWER_LEVEL or f1 == 1:
                        doexit = True
                elif jump == 1 or expr_opt == win_expr:
                    doexit = True
                elif ct_now == 0:
                    print("#################################")
                    print("")
                    print("SORRY NO SOLUTION FOUND")
                    print(str(sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])) + "/" + str(len(res)), " records matched")
                    print("")
                    print("#################################")
                    print("")
                    return None, None
                else:
                    jump = int(jump/2)
                    ct_now = ct_now - jump
                
            if doexit:
                print("")
                print("#################################")
                print("")
                print("OPTIMUM POWER LEVEL is", ct_opt)
                print("")
                print(f"Precision: {opt_precision_sofar * 100:.2f}%")
                print(f"Recall: {opt_recall_sofar * 100:.2f}%")
                print(f"F1 Score: {opt_f1_sofar * 100:.2f}%")
                print(f"Effectiveness & Efficiency Score: {best_ee_sofar * 100:.3f}%")
                print("Expression:")
                print(self.replaceSegName(expr_opt))
                print("")
                print("#################################")
                print("")
        
                self.expression_opt = expr_opt
                self.opt_f1 = f1
                
                return ct_opt

    def optimize_compact(self, test_data, answer, cnt_out=20, useUnion=False):

        inp = test_data
        
        print("Analysis started")
        false_clauses = sorted([(v, k) for k, v in self.false_confidence.items()], reverse=True)
        true_clauses = sorted([(v, k) for k, v in self.true_confidence.items()], reverse=True)
        
        all_clauses = sorted([(v, k) for k, v in self.all_confidence.items()], reverse=True)
        
#         print(len(true_clauses), "true clauses")
#         print(len(false_clauses), "false clauses")
#         print(len(all_clauses), "all clauses")
        
        final_expr = ""
        best_ee = -1
        
        true_exps = []
        false_exps = []
        if len(all_clauses) > 0:
            
            cnt = 0
            for i in range(len(all_clauses)):
#             for i in range(1, len(all_clauses), 1):
                cnt = cnt + 1
                if cnt_out < cnt:
                    break
                    
                if i % 10 == 0:
                    print(str(i) + "/" + str(len(all_clauses)) + " completed" )

                expr = ""
                temp_true_exps = copy.deepcopy(true_exps)
                temp_false_exps = copy.deepcopy(false_exps)
                if all_clauses[i][1] in self.true_confidence:
                    temp_true_exps.append(all_clauses[i][1])
                else:
                    temp_false_exps.append(all_clauses[i][1])
                    
                true_expression = "(" + " | ".join(temp_true_exps) + ")"
                false_expression = "not (" + " | ".join(temp_false_exps) + ")"
            
                connector = " | " if useUnion else " & "
        
                if len(temp_true_exps) > 0:
                    expr = true_expression 
                if len(temp_true_exps) > 0 and len(temp_false_exps) > 0:
                    expr = expr + connector
                if len(temp_false_exps) > 0:
                    expr = expr + false_expression
                    
                expr = expr.replace(" & ", " and ").replace(" | ", " or ")
                res = self.solve_direct(inp, expr)
                precision = precision_score(answer, res)
                recall = recall_score(answer, res)
                f1 = f1_score(answer, res)
                ee = (f1 + min(precision,recall))/2
                ee = 0 if ee < 0 else ee
                
                if i == 0:
                    final_expr = expr
                
                    true_exps = temp_true_exps
                    false_exps = temp_false_exps
                    
                if best_ee < ee:
                    best_ee = ee

                    final_expr = expr
                
                    true_exps = temp_true_exps
                    false_exps = temp_false_exps
                
                    cnt = 0


        if final_expr != "":
            
            print("Assessment of the optimal solution")
            res = self.solve_direct(inp, final_expr)

            if len(res) > 0:
                print("")
                print("#### DERIVED OPTIMUM EXPRESSION ####")
                print("")
                precision = precision_score(answer, res)
                recall = recall_score(answer, res)
                f1 = f1_score(answer, res)
                print(str(sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])) + "/" + str(len(res)), " records matched " + f" ({sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])/len(res)*100:.2f}%)")
                print(f"Precision: {precision * 100:.2f}%")
                print(f"Recall: {recall * 100:.2f}%")
                print(f"F1 Score: {f1 * 100:.2f}%")

                self.opt_f1 = f1
                
                print(self.replaceSegName(final_expr))
                
                print("")
                
                self.expression_opt = final_expr
                
                return final_expr
            
            
    def random_split_matrix(matrix, divide_by=2):
        matrix = copy.deepcopy(matrix)
        rows = list(matrix)  # Convert to list for easier shuffling
        random.shuffle(rows)  # Shuffle rows in-place
        split_index = len(rows) // divide_by  # Integer division for equal or near-equal halves
        return rows[:split_index], rows[split_index:]

    def train_and_optimize(self, data_list=None, max_dnf_len=4, error_tolerance=0.00, 
                       min_match=0.00, use_approx_dnf=False, redundant_thresh=1.00, elements_count_penalty=1.0, 
                           use_compact_opt=False, cnt_out=20, useUnion=False, useExpanded=True, use_stochastic=False, stochastic_min_rating=0.97):
        
        print("Training started...")
        
        headers = data_list[0]
        data_list2 = data_list[1:]
        
        train_data, valid_data = Deterministic_Regressor.random_split_matrix(data_list2)

        train_inp = [headers] + train_data
        
        self.train(data_list=train_inp, max_dnf_len=max_dnf_len, 
                        error_tolerance=error_tolerance, min_match=min_match, use_approx_dnf=use_approx_dnf, redundant_thresh=redundant_thresh, 
                   useExpanded=useExpanded, use_stochastic=use_stochastic, stochastic_min_rating=stochastic_min_rating)

        print("Optimization started...")
        inp = [headers] + valid_data
           
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]
                
        answer = [int(inp[i][-1]) for i in range(1, len(inp), 1)]
        inp = [row[:-1] for row in inp]
             
        if use_compact_opt:
            return self.optimize_compact(inp, answer, cnt_out=cnt_out, useUnion=useUnion)
        else:
            return self.optimize_params(inp, answer, elements_count_penalty=1.0, useUnion=useUnion)
    
    def train_and_optimize_bulk(self, data_list, expected_answers, max_dnf_len=4, error_tolerance=0.02,  
                   min_match=0.03, use_approx_dnf=False, redundant_thresh=1.00, elements_count_penalty=1.0, use_compact_opt=False, cnt_out=20, useUnion=False, 
                                useExpanded=True, use_stochastic=False, stochastic_min_rating=0.97):

        self.children = [Deterministic_Regressor() for _ in range(len(expected_answers))]
        
        cnt_recs = len(expected_answers[0])

        for i in range(len(self.children)):
            if self.classDic is not None and len(self.classDic) > 0:
                print("")
                print("=====================================================================================")
                print("Start training class", self.classDic[i], "(" + str(self.item_counts[self.classDic[i]]) + "/"+ str(cnt_recs) + ")")
                print("")
            else:
                print("Child", i)
            
            self.children[i].dic_segments = copy.deepcopy(self.dic_segments)
            d_list = copy.deepcopy(data_list)
            d_list[0].append("res")
            
            for k in range(len(d_list)-1):
                d_list[k+1].append(expected_answers[i][k])
            self.children[i].train_and_optimize(data_list=d_list, max_dnf_len=max_dnf_len, error_tolerance=error_tolerance, 
                    min_match=min_match, use_approx_dnf=use_approx_dnf, redundant_thresh=redundant_thresh, 
                        elements_count_penalty=elements_count_penalty, use_compact_opt=use_compact_opt, cnt_out=cnt_out, useUnion=useUnion, 
                        useExpanded=useExpanded, use_stochastic=use_stochastic, stochastic_min_rating=stochastic_min_rating)

            
            
    def solve_with_opt_bulk(self, inp_p):
        res = []
        for c in self.children:
            r = c.solve_with_opt(inp_p)
            if r == None or len(r) == 0:
                r == [0] * (len(inp_p)-1)
            res.append(r)
            
        return res
    
    def solve_with_opt_class(self, inp_p):
        
        res = self.solve_with_opt_bulk(inp_p)
        
        dic_f1 = {i: self.children[i].opt_f1 for i in range(len(self.children))}
        
        len_rows = len(res[0])
        len_res = len(res)
        new_res = [0] * len_rows
        for i in range(len_rows):
            numbers = [s[1] for s in sorted([(random.random()*dic_f1[i], i) for i in range(len_res)], reverse=True)]
            for k in range(len(numbers)):
                if res[numbers[k]][i] == 1:
                    new_res[i] = self.classDic[numbers[k]]
                    break
                if k == len(numbers) - 1:
                    new_res[i] = self.classDic[numbers[k]]
                    

        return new_res
    
    def solve_with_opt_continuous(self, inp_p, inp_p_org):
        
#         res = self.solve_with_opt_bulk(self.get_test_dat_with_head())
        res = self.solve_with_opt_bulk(inp_p)

        dic_f1 = {i: self.children[i].opt_f1 for i in range(len(self.children))}

        len_rows = len(res[0])
        len_res = len(res)
        new_res = [0] * len_rows
        
#         inp_p_org_np = np.array(inp_p_org)
        
        winners = []
        for i in range(len_rows):
            numbers = [s[1] for s in sorted([(random.random()*dic_f1[i], i) for i in range(len_res)], reverse=True)]
            for k in range(len(numbers)):
                if res[numbers[k]][i] == 1:
                    cl = self.classDic[numbers[k]]
                    new_res[i] = self.predictors[cl].predict([[c for i, c in enumerate(inp_p_org[i]) if i in self.combo_list[cl]]])[0]
                    winners.append(cl)
                    break
                if k == len(numbers) - 1:
                    cl = reg.classDic[numbers[k]]
                    winners.append(cl)
                    new_res[i] = self.predictors[cl].predict([[c for i, c in enumerate(inp_p_org[i]) if i in self.combo_list[cl]]])[0]

        count_dict = Counter(winners)
        for f in sorted([(v, k) for k, v in count_dict.items()], reverse=True):
            print(f"{reg.combo_list[f[1]]} occurs {f[0]} times.")

        return new_res
    
    def solve_with_highest(self, inp_p_org):
        
        new_res = [0] * len(inp_p_org)
        print(self.combo_list[0], "is used")
        for i in range(len(inp_p_org)):
            
            new_res[i] = self.predictors[0].predict([[c for i, c in enumerate(inp_p_org[i]) if i in self.combo_list[0]]])[0]
            
        return new_res
    
    
    def train_and_optimize_class(self, data_list, expected_answers, max_dnf_len=4, error_tolerance=0.00, 
               min_match=0.00, use_approx_dnf=False, redundant_thresh=1.00, elements_count_penalty=1.0, use_compact_opt=False, cnt_out=20, useUnion=False, 
                                 useExpanded=True, use_stochastic=False, stochastic_min_rating=0.97):

        # Use Counter to perform group-by count
        cnt_recs = len(expected_answers)
        self.item_counts = Counter(expected_answers)
        classList = sorted([item for item, count in self.item_counts.items()])
        classList = [(i, classList[i]) for i in range(len(classList))]
        self.classDic = {c[0]: c[1] for c in classList}
        self.classDicRev = {c[1]: c[0] for c in classList}
        answers = [[0 for _ in range(len(expected_answers))] for _ in range(len(classList))]
        for i in range(len(answers[0])):
            answers[self.classDicRev[expected_answers[i]]][i] = 1

        self.train_and_optimize_bulk(data_list=data_list, expected_answers=answers, max_dnf_len=max_dnf_len, error_tolerance=error_tolerance, 
                    min_match=min_match, use_approx_dnf=use_approx_dnf, redundant_thresh=redundant_thresh, 
                                     elements_count_penalty=elements_count_penalty, use_compact_opt=use_compact_opt, cnt_out=cnt_out, useUnion=useUnion, 
                                     useExpanded=useExpanded, use_stochastic=use_stochastic, stochastic_min_rating=stochastic_min_rating)
    
    def prepropcess(self, whole_rows, by_two, splitter=3, add_cluster_label=False):
        
        whole_rows_copy = copy.deepcopy(whole_rows)
        if add_cluster_label:
            print("Adding cluster label")
            head = whole_rows_copy[0:1]
            data = whole_rows_copy[1:]
            data = data.tolist() if isinstance(data, np.ndarray) else copy.deepcopy(data)
            clusters, self.gmm = Deterministic_Regressor.findClusters(data)
            if clusters is None or len(clusters) == 0:
                print("No cluster found")
            else:
#                 count_dict = Counter(clusters)
#                 for f in sorted([(v, k) for k, v in count_dict.items()], reverse=True):
#                     print(f"Cluster {f[1]} occurs {f[0]} times.")
    
#                 print(len(clusters), "clusters found")
                for i, row in enumerate(data):
                    row.insert(-1, clusters[i])
                head[0].insert(-1, "clustered_label")
                whole_rows_copy = head + data 
    #             print("data[:3]", data[:3])
            
#         self.whole_rows = self.clean_and_discretize(whole_rows, by_two)
        self.whole_rows = self.clean_and_discretize(whole_rows_copy, by_two)
        headers = self.whole_rows
        data = self.whole_rows[1:]
        random.shuffle(data)  # Shuffle rows in-place
        split_index = len(data) // splitter  # Integer division for equal or near-equal halves
        self.test_rows = data[:split_index]
        self.train_rows = data[split_index:]
    
    def prepropcess_continous(self, whole_rows, by_two, splitter=3, max_reg=2, thresh=0.3, add_quads=False, max_vars=3, omit_similar=False, include_all=True, include_related=True, sample_limit=0, num_fit=5, 
                              use_multinomial=False, add_cluster_label=False, use_piecewise=True):
        whole_rows_org = copy.deepcopy(whole_rows)
        headers_org = whole_rows_org[0]
        data_org = whole_rows_org[1:]
        
        if add_cluster_label:
            print("Adding cluster label")
            data_org = data_org.tolist() if isinstance(data_org, np.ndarray) else copy.deepcopy(data_org)
            clusters, self.gmm = Deterministic_Regressor.findClusters(data_org)
            if clusters is None or len(clusters) == 0:
                print("No cluster found")
            else:
                count_dict = Counter(clusters)
#                 for f in sorted([(v, k) for k, v in count_dict.items()], reverse=True):
#                     print(f"Cluster {f[1]} occurs {f[0]} times.")
                for i, row in enumerate(data_org):
                    row.insert(-1, clusters[i])
                headers_org.insert(-1, "clustered_label")
#                 print("headers_org", headers_org)
#                 print("data_org[:3]", data_org[:3])
        
        random.shuffle(data_org)  # Shuffle rows in-place
#         print("headers_org", headers_org)
#         print("data_org[:3]", data_org[:3])
        
        self.whole_rows_org = [headers_org] + data_org
#         print("self.whole_rows_org[:3]", self.whole_rows_org[:3])
        self.whole_rows = self.clean_and_discretize(self.whole_rows_org, by_two)
#         print("self.whole_rows[:3]", self.whole_rows[:3])
        
        headers = self.whole_rows[0]
        data = self.whole_rows[1:]

        split_index = len(data) // splitter  # Integer division for equal or near-equal halves
        self.test_rows = data[:split_index]
        self.train_rows = data[split_index:]
        
        self.test_rows_org = data_org[:split_index]
        self.train_rows_org = data_org[split_index:]
        if add_quads:
            target_cols = [i for i in range(len(data_org[0])-1) if Deterministic_Regressor.IsNonBinaryNumeric([row[i] for row in data_org])]
            X = copy.deepcopy(self.test_rows_org)
            for j in range(len(X[0])-1):
                if j not in target_cols:
                    continue
                for i, xx in enumerate(X):
#                     self.test_rows_org[i].insert(-1, X[i][j]**2)
                    self.test_rows_org[i].insert(-1, X[i][j]**2 if X[i][j] >= 0 else (X[i][j]**2)*-1)
            for j in range(len(X[0])-1):
                if j not in target_cols:
                    continue
                for i, xx in enumerate(X):
#                     self.test_rows_org[i].insert(-1, np.sqrt(np.abs(X[i][j])))
                    self.test_rows_org[i].insert(-1, np.sqrt(X[i][j]) if X[i][j] >= 0 else np.sqrt(X[i][j]*-1)*-1)
                    
            X = copy.deepcopy(self.train_rows_org)
            for j in range(len(X[0])-1):
                if j not in target_cols:
                    continue
                for i, xx in enumerate(X):
#                     self.train_rows_org[i].insert(-1, X[i][j]**2)
                    self.train_rows_org[i].insert(-1, X[i][j]**2 if X[i][j] > 0 else (X[i][j]**2)*-1)
            for j in range(len(X[0])-1):
                if j not in target_cols:
                    continue
                for i, xx in enumerate(X):
#                     self.train_rows_org[i].insert(-1, np.log(X[i][j]) if not np.isnan(np.log(X[i][j])) else 0)
#                     log_val = 0. if X[i][j] < 1 else np.log(X[i][j])
#                     if np.isnan(log_val) or log_val == float("-inf") or log_val == float("inf"):
#                         log_val = 0.
#                     self.train_rows_org[i].insert(-1, log_val)
#                     val = np.sqrt(np.abs(X[i][j]))
#                     if np.isnan(val):
#                         val = 0
#                     self.train_rows_org[i].insert(-1, np.sqrt(np.abs(X[i][j])))
                    self.train_rows_org[i].insert(-1, np.sqrt(X[i][j]) if X[i][j] >= 0 else np.sqrt(X[i][j]*-1)*-1)
    
#         print("self.get_train_dat_org_wo_head()[:20]", self.get_train_dat_org_wo_head()[:20])
        return self.continuous_regress(self.get_train_dat_org_wo_head(), self.get_train_res_org_wo_head(), max_reg=max_reg, thresh=thresh, max_vars=max_vars, omit_similar=omit_similar, 
                                       include_all=include_all, include_related=include_related, sample_limit=sample_limit, num_fit=num_fit, use_multinomial=use_multinomial, use_piecewise=use_piecewise)
    
    def get_train_dat_wo_head(self):
        return [row[:-1] for row in self.train_rows]
    def get_train_res_wo_head(self):
        return [row[-1] for row in self.train_rows]
    def get_train_dat_with_head(self):
        return [self.whole_rows[0][:-1]] + [row[:-1] for row in self.train_rows]
    def get_train_datres_wo_head(self):
        return self.train_rows
    def get_train_datres_with_head(self):
        return [self.whole_rows[0]] + self.train_rows
    def get_test_dat_wo_head(self):
        return [row[:-1] for row in self.test_rows]
    def get_test_res_wo_head(self):
        return [row[-1] for row in self.test_rows]
    def get_test_dat_with_head(self):
        return [self.whole_rows[0][:-1]] + [row[:-1] for row in self.test_rows]
    def get_test_datres_wo_head(self):
        return self.test_rows[1:]
    def get_test_datres_with_head(self):
        return [self.whole_rows[0]] + self.test_rows
    
    def get_train_dat_org_wo_head(self):
        return [row[:-1] for row in self.train_rows_org]
    def get_train_res_org_wo_head(self):
        return [row[-1] for row in self.train_rows_org]
#     def get_train_dat_org_with_head(self):
#         return [self.whole_rows_org[0][:-1]] + [row[:-1] for row in self.train_rows_org]
    def get_train_datres_org_wo_head(self):
        return self.train_rows_org
#     def get_train_datres_org_with_head(self):
#         return [self.whole_rows_org[0]] + self.train_rows_org
    def get_test_dat_org_wo_head(self):
        return [row[:-1] for row in self.test_rows_org]
    def get_test_res_org_wo_head(self):
        return [row[-1] for row in self.test_rows_org]
#     def get_test_dat_org_with_head(self):
#         return [self.whole_rows_org[0][:-1]] + [row[:-1] for row in self.test_rows_org]
    def get_test_datres_org_wo_head(self):
        return self.test_rows_org[1:]
#     def get_test_datres_org_with_head(self):
#         return [self.whole_rows_org[0]] + self.test_rows_org
    
    def show_stats(predicted, actual, average="weighted", elements_count_penalty=1.0):
        
        if len(predicted) != len(actual):
            print("The row number does not match")
            return
        answer = actual
        res = predicted
        precision = precision_score(answer, res, average=average, labels=np.unique(res))
        recall = recall_score(answer, res, average=average, labels=np.unique(res))
        f1 = f1_score(answer, res, average=average, labels=np.unique(res))
        print("")
        print("####### PREDICTION STATS #######")
        print("")
        print(str(sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])) + "/" + str(len(res)), " records matched" + f" ({sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])/len(res)*100:.2f}%)")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")
        print("")
        print("##############")
        print("")


    def show_mse(y_test, y_pred):
        #         # The mean squared error
        print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
        
    def continuous_regress(self, X_train, y_train, X_test=None, y_test=None, max_reg=2, thresh=0.3, max_vars=3, omit_similar=False, include_all=True, include_related=True, sample_limit=0, num_fit=5, 
                           use_multinomial=False, use_piecewise=True):

    #     if test_size == 0.0:
    #         X_train, X_test, y_train, y_test = X, X, y, y
    #     else:
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        if X_test is None:
            X_test = X_train
            y_test = y_train
            
#         print("y_train", y_train)
        is_logistic = all([y == 0 or y == 1 for y in y_train])

        if is_logistic:
            print("Logistic regression is used")
                    
        target_cols = [i for i, xx in enumerate(X_train[0]) if Deterministic_Regressor.IsNonBinaryNumeric([row[i] for row in X_train])]
        print("target_cols", target_cols)
        self.target_cols = target_cols
        X = np.array([np.array(x) for x in X_train])
        X_test = np.array([np.array(x) for x in X_test])

        numbers = list(range(len(X_train[0])))

        col_to_ignore = Deterministic_Regressor.give_highly_correlated_columns(X, target_cols)

        dic_errors_all = {}
        dic_sme = {}
        dic_ind = {}
        predictors = {}
        # Generate all combinations of two numbers from the list

        all_best_sme = float("inf")
        
        print("Now regressing: ", sep=' ', end='')
        if include_all:
            combo = tuple(target_cols)
            print(combo, sep=' ', end='')
            print(" ", end='')
#             model = LinearRegression()
            if use_multinomial:
#                 model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                model = piecewise_regressor(regression_type="multi") if use_piecewise else LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            elif is_logistic:
#                 model = LogisticRegression(solver='lbfgs', max_iter=1000)
                model = piecewise_regressor(regression_type="logistic") if use_piecewise else LogisticRegression(solver='lbfgs', max_iter=1000)
            else:
#                 model = LinearRegression() if len(combo) < 10 else ElasticNet()
#                 model = LinearRegression()
                model = piecewise_regressor(regression_type="linear") if use_piecewise else LinearRegression()
            model.fit(X[:, combo], y_train)
            y_pred = model.predict(X_test[:, combo])
            dic_errors_all[combo] = np.abs(y_pred - y_test)
            predictors[combo] = model
            all_best_sme = mean_squared_error(y_test, y_pred) if not use_multinomial else 1 - accuracy_score(y_test, y_pred)
            dic_sme[combo] = all_best_sme

        ind_all = 0
        numloop = max_vars if len(target_cols) > max_vars else len(target_cols)
        model = None
        for i in range(numloop):
            combinations_of_two = list(combinations(numbers, i+1))
            print("")
            print("Regressing", (i+1), "variable combos")
            # Print the combinations
            for combo in combinations_of_two:
                dic_ind[ind_all] = combo
                ind_all += 1

                if any([c in col_to_ignore for c in combo]):
                    continue
                if not all([c in target_cols for c in combo]):
                    continue
                if combo in dic_sme:
                    continue
                    
#                 print(combo, sep=' ', end='')
#                 print(" ", end='')
                print(".", end='')
                if sample_limit == 0:
    
                    if use_multinomial:
                        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                    elif is_logistic:
                        model = LogisticRegression(solver='lbfgs', max_iter=1000)
                    else:
#                         model = LinearRegression() if len(combo) < 10 else ElasticNet()
                        model = LinearRegression()

                    # Train the model using the training sets
                    model.fit(X[:, combo], y_train)
                    # The coefficients
                    # The mean squared error
                    # Make predictions using the testing set
                    y_pred = model.predict(X_test[:, combo])
                    the_sme = mean_squared_error(y_test, y_pred) if not use_multinomial else 1 - accuracy_score(y_test, y_pred)
                else:
                    best_sme = float("inf")
                    for i in range(num_fit):
                        # Create a linear regression model
                        tmp_model = LinearRegression()
                        if use_multinomial:
                            tmp_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                        elif is_logistic:
                            tmp_model = LogisticRegression(solver='lbfgs', max_iter=1000)
                        else:
#                             tmp_model = LinearRegression() if len(combo) < 5 else ElasticNet()
                            tmp_model = LinearRegression()

    
                        # Train the model using the training sets
                        tmp_model.fit(X[:, combo], y_train)

                        sampled = random.sample(list(range(len(X))), sample_limit if len(X) > sample_limit else len(X))
                        tmp_model.fit(X[:, combo][sampled,:], np.array(y_train)[sampled])

                        # Make predictions using the testing set
                        y_pred = tmp_model.predict(X_test[:, combo])
                        tmp_sme = mean_squared_error(y_test, y_pred) if not use_multinomial else 1 - accuracy_score(y_test, y_pred)
                        if best_sme > tmp_sme:
                            best_sme = tmp_sme
                            model = tmp_model

                    the_sme = best_sme
                    
                dic_sme[combo] = the_sme
                dic_errors_all[combo] = np.abs(y_pred - y_test)
                predictors[combo] = model
        
        print("")
        i = 0
        already_set = set()
        already_list = list()
        min_sme = -1
        already_tup = ()
            
        for f in sorted([(v, k) for k, v in dic_sme.items()]):
            if i == 0:
                min_sme = f[0]
            elif min_sme+min_sme*thresh < f[0]:
                break
            if i == max_reg:
                break
            if omit_similar and not include_all and any([sum([1 if k in a else 0 for k in f[1]]) > len(f[1]) - 2 for a in already_set]):
                    continue
            already_set.add(f[1])
            print("key", f[1], "sme", f[0])
            already_list.append(f[1])
            i += 1


#         print("already_list", already_list)
        if include_related:
            numtop = 8
            related_cols = []
#             for a in already_list:
            for f in sorted([(v, k) for k, v in dic_sme.items()]):
                if len(f[1]) > 3:
                    continue
                for aa in f[1]:
#                     print("aa", aa)
                    if aa not in related_cols:
                        related_cols.append(aa)
                    if len(related_cols) >= numtop:
                        break
                
                if len(related_cols) >= numtop:
                    break
#             print("related_cols", related_cols)
            if len(related_cols) > 1:
                related_cols = tuple(sorted(related_cols))
                if related_cols not in already_list:
#                     related_cols = Deterministic_Regressor.give_correlated_columns_to_y(X_train, y_train, threshold=0.5)
#                     if len(related_cols) > 0 and tuple(target_cols) != tuple(related_cols):
#                     combo = tuple(related_cols)
                    combo = related_cols
#                     print(combo, sep=' ', end='')
#                     print(" ", end='')
                    if use_multinomial:
#                         model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                        model = piecewise_regressor(regression_type="multi") if use_piecewise else LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                    elif is_logistic:
#                         model = LogisticRegression(solver='lbfgs', max_iter=1000)
                        model = piecewise_regressor(regression_type="logistic") if use_piecewise else LogisticRegression(solver='lbfgs', max_iter=1000)
                    else:
#                         model = LinearRegression()
                        model = piecewise_regressor(regression_type="linear") if use_piecewise else LinearRegression()
                    model.fit(X[:, combo], y_train)
                    y_pred = model.predict(X_test[:, combo])
                    dic_errors_all[combo] = np.abs(y_pred - y_test)
                    predictors[combo] = model
                    the_sme = mean_squared_error(y_test, y_pred) if not use_multinomial else 1 - accuracy_score(y_test, y_pred)
                    print("key", combo, "sme", the_sme)
                    dic_sme[combo] = the_sme
                    already_list.append(combo)

        winner_predictors = [predictors[a] for i, a in enumerate(already_list)]
        self.combo_list = already_list
        self.predictors = winner_predictors

        winners = [0] * len(y_test)
        for i, y in enumerate(y_test):
            lowest = float("inf")
            lowest_ind = float("inf")
            for j, a in enumerate(already_list):
                if dic_errors_all[a][i] < lowest:
                    lowest_ind = j
                    lowest = dic_errors_all[a][i]
            winners[i] = lowest_ind

        return winners
    
    def show_regression_info(self):
        for i, p in enumerate(self.predictors):
            print("")
            print("Regression Variables:", self.combo_list[i])
            print("Coefficients:", p.coef_)
            print("Intercept:", p.intercept_)
            print("----------")