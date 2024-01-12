# Name: DNF_Regression_solver
# Author: tomio kobayashi
# Version: 2.2.7
# Date: 2024/01/12

import itertools
from sympy.logic import boolalg
import numpy as np

import sklearn.datasets
import pandas as pd
import random
from sympy import simplify

class DNF_Regression_solver:
# This version has no good matches
# Instead, all true matches are first added, and some are removed when 
# false unmatch exists and there is no corresponding other rule
    def __init__(self):
        self.expression = ""
        self.expression_common = ""
        self.expression_true = ""
        self.expression_false = ""
        self.expression_union = ""
        
    def cnf_to_dnf(cnf):
        dnf_clauses = []

        for clause in cnf:
            dnf_clause = []
            for literal in clause:
                if isinstance(literal, tuple):  # Handle negation
                    dnf_clause.append((literal[0], not literal[1]))
                else:
                    dnf_clause.append(literal)
            dnf_clauses.append(dnf_clause)

        dnf_result = []
        if len(dnf_clauses) > 0:
            for i in range(len(dnf_clauses[0])):
                dnf_result.append(sorted(list(set([dnf_clauses[j][i] for j in range(len(dnf_clauses))]))))

        return dnf_result

    def try_convert_to_numeric(text):
        for func in (int, float, complex):
            try:
                return func(text)
            except ValueError:
                pass

        return text  # Return the original string if conversion fails

    def convTuple2bin(t, width):
        i = 1
        ret = 0
        for x in t:
            ii = i << width - x - 1
            ret = ret | ii
        return ret

    #     This needs to be separate function because the condition 
    #     variables cannot have conflict with function variables.  Thus "__XXX__" format is used to prevent naming conflicts.    
#     def myeval(__ccc___, __tokens___, __inp___):
#     def myeval(self, __ccc___, __tokens___):
#         for __jjjj___ in range(len(__tokens___)):
#             exec(__tokens___[__jjjj___] + " = " + str(__ccc___[__jjjj___]))
#         return eval(self.expression)
            
    def myeval(__ccc___, __tokens___, ______s____):
        for __jjjj___ in range(len(__tokens___)):
            exec(__tokens___[__jjjj___] + " = " + str(__ccc___[__jjjj___]))
        return eval(______s____)
            
        
    def solve(self, inp_p, check_negative=True, used_expression=""):
#         print("SOLVE")
        
        inp = copy.deepcopy(inp_p)
        
#         print("inp[0]", inp[0])
        numvars = len(inp[0])

        if check_negative:
            for i in range(numvars):
                inp[0].insert(i+numvars, "n_" + inp[0][i])
#                 inp[0].insert(i+numvars-1, "n_" + inp[0][i])
            for j in range(1, len(inp), 1):
                for i in range(numvars):
                    inp[j].insert(i+numvars,"0" if inp[j][i] == "1" else "1")
#                     inp[j].insert(i+numvars-1,"0" if inp[j][i] == "1" else "1")
            numvars *= 2
        
        
#         tokens = [inp[0][i] for i in range(len(inp[0])-1)]
#         tokens = [inp[0][i] for i in range(numvars-1)]
#         tokens = inp[0][:-1]
        tokens = inp[0]
#         print("tokens", tokens)
#         inp_list = np.array([np.array([inp[i][j] for j in range(len(inp[i]))]) for i in range(1, len(inp), 1)])
#         inp_list = [[inp[i][j] for j in range(len(inp[i]))] for i in range(1, len(inp), 1)]
#         inp_list = [[inp[i][j] for j in range(len(inp[i]))] for i in range(1, len(inp), 1)]
#         inp_list = [row[:-1] for row in inp[1:]]
        inp_list = [row for row in inp[1:]]
        res = list(range(len(inp_list)))
        
        if used_expression == "" or used_expression == "common":
            self.expression = self.expression_common
        elif used_expression == "true":
            self.expression = self.expression_true
        elif used_expression == "false":
            self.expression = self.expression_false
        elif used_expression == "union":
            self.expression = self.expression_union

        
#         print("self.expression_common", self.expression_common)
#         print("self.expression_true", self.expression_true)
#         print("self.expression_false", self.expression_false)
#         print("self.expression_union", self.expression_union)
        
        
        if self.expression == "":
            print("The expression is not available")
            return []
        
        print("Solver Expression:")
        print(self.expression)
        for i in range(len(inp_list)):
#             res[i] = self.myeval(inp_list[i], tokens)
            res[i] = DNF_Regression_solver.myeval(inp_list[i], tokens, self.expression)

        return res

    def generate_segment_ranks(df, num_segments, name):
    #     df = pd.DataFrame({name: data})
    #     print("df", df)
        df[name + '_rank'] = pd.cut(df[name], bins=num_segments, labels=False)
        df[name + '_label'] = pd.cut(df[name], bins=num_segments, labels=[f'{name} {i+1}' for i in range(num_segments)])
        min_max_per_group = df.groupby(name + '_rank')[name].agg(['max'])
        print("")
        print(min_max_per_group)
        return df

    def discretize_data(data_list, by_four=1):

    #     lst = [['apple', 'red', 11], ['grape', 'green', 22], ['orange', 'orange', 33], ['mango', 'yellow', 44]] 
    #     data = pd.DataFrame(values, columns =headers, dtype = float) 
    
        result_header = data_list[0][-1]
        result_values = [d[-1] for d in data_list[1:]]
        
        headers = data_list[0][:-1]
        values = [d[:-1] for d in data_list[1:]]

#         print("result_header", result_header)
#         print("result_values", result_values)
#         print("headers", headers)
#         print("values", values)
    
        data = pd.DataFrame(values, columns=headers) 

#         print("data", data)
        cols = [c for c in data.columns]
        for c in cols:
            countNonBool = len(data[c]) - (data[c] == 0).sum() - (data[c] == 1).sum()
            if countNonBool > 0 and pd.api.types.is_numeric_dtype(data[c]):
                result_df = DNF_Regression_solver.generate_segment_ranks(data, by_four*4, c)
                one_hot_df = pd.get_dummies(result_df[c + '_rank'], prefix=c)
                one_hot_df = one_hot_df.astype(int)
                data = pd.concat([result_df, one_hot_df], axis=1)

#         print("data", data)
        # data = data.filter(lambda col: (col == 0).all() or (col == 1).all())
        # data = data.loc[:, (data == 0) | (data == 1).any()]

        cols = [c for c in data.columns]
#         print("cols", cols)
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

        return sampled_rows

    
    def train(self, file_path=None, data_list=None, max_dnf_len=4, check_false=True, check_negative=False, error_tolerance=0.02, by_four=1):

# file_path: input file in tab-delimited text
# check_negative: enable to check the negative conditions or not.  This one is very heavy.
# max_dnf_len: max length of AND clauses in output DNF.  
#       Increasing this one is heavey especially if check_negative is True
# drop_fake: enable to drop the clause that met the true condition, but not false condition.  This one is heavy.

        # Example usage:
        # file_path = '/kaggle/input/dnf-regression/dnf_regression.txt'
        # file_path = '/kaggle/input/tomio5/dnf_regression.txt'

        inp = None
        if file_path is not None:
            with open(file_path, 'r') as f:
                inp = [line.strip().split('\t') for line in f]
        else:
            inp = data_list

# ############## TO BE REMOVED ############## 
        print("num recs before", len(inp))
        inp = DNF_Regression_solver.reduce_rows_except_first(inp, 20)
        print("num recs after", len(inp))
# ############## TO BE REMOVED ############## 


        inp = [[DNF_Regression_solver.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]
        
#         print("inp", inp)
#         print("len(inp[1])", len(inp[1]))
        print("Discretizing...")
        inp = DNF_Regression_solver.discretize_data(inp, by_four)
        print("")
        
        numvars = len(inp[1])-1

#         print("len(inp[1])", len(inp[1]))
#         print("check_negative", check_negative)
        
        if check_negative:
            for i in range(numvars):
                inp[0].insert(i+numvars, "n_" + inp[0][i])
            for j in range(1, len(inp), 1):
                for i in range(numvars):
                    inp[j].insert(i+numvars, 0 if inp[j][i] == 1 else 1)
            numvars *= 2

        print("Columns:")
        print(inp[0])
        print("")
        
        if max_dnf_len > numvars - 1:
            max_dnf_len = numvars - 1
            
        dic = dict()
                    
        dic_opp = dict()
        
        true_list = []
        false_list = []
        for i in range(1, len(inp), 1):
            s = ""
            for j in range(len(inp[i]) - 1):
                s += str(inp[i][j])
#             print(s)
            truefalse = inp[i][len(inp[i]) - 1]
            dic[int(s, 2)] = truefalse
            if truefalse == '1':
                true_list.append(s)
            else:
                false_list.append(s)

        inp_oppo = [copy.deepcopy(inp[0])] + [[0 if m == 1 else 1 for m in inp[i]] for i in range(1, len(inp), 1)]
        for i in range(1, len(inp_oppo), 1):
            s = ""
            for j in range(len(inp_oppo[i]) - 1):
#                 s += inp[i][j]
                s += str(inp_oppo[i][j])
            truefalse = inp_oppo[i][len(inp_oppo[i]) - 1]
            dic_opp[int(s, 2)] = truefalse

                
        print("Deriving true expressions...")
        dnf_perf = list()
        raw_perf = list()
        raw_perf2 = list()
        for s in range(max_dnf_len):
            len_dnf = s + 1
            
            l = [ii for ii in range(numvars)]
            p_list = list(itertools.combinations(l, len_dnf))
            print(str(len_dnf) + " variable patterns")
            if len(p_list) > 1000000:
                print("Skipping because " + str(len(p_list)) + " combinations is too many")
                break
            true_test_pass = True
            for i in range(len(p_list)):
                match_and_break = False
                b = DNF_Regression_solver.convTuple2bin(p_list[i], numvars)
                for p in raw_perf2:
                    if p == b & p:
                        match_and_break = True
                        break
                if match_and_break:
                    continue
                r = [v for k,v in dic.items() if k & b == b]
                if len(r) == 0:
                    continue
                cnt_all = len([f for f in r])
                cnt_unmatch = len([f for f in r if f == 0])
                if cnt_unmatch/cnt_all > error_tolerance:
#                 if cnt_unmatch > 0:
                    continue

                raw_perf.append([ii for ii in p_list[i]])
                raw_perf2.append(b)
        
        for dn in raw_perf:
            dnf_perf.append(sorted(list(set([inp[0][ii] for ii in dn]))))

        print("size of true dnf " + str(len(dnf_perf)))
        
        print("Deriving false expressions...")
        dnf_perf_n = list()
        raw_perf_n = list()
        raw_perf2_n = list()
        if check_false:
            if check_negative:
                for s in range(max_dnf_len):
                    len_dnf = s + 1
                    l = [ii for ii in range(numvars)]
                    p_list = list(itertools.combinations(l, len_dnf))
                    print(str(len_dnf) + " variable patterns")
                    if len(p_list) > 1000000:
                        print("Skipping because " + str(len(p_list)) + " combinations is too many")
                        break

                    true_test_pass = True
                    for i in range(len(p_list)):
                        match_and_break = False
                        b = DNF_Regression_solver.convTuple2bin(p_list[i], numvars)
                        for p in raw_perf2_n:
                            if p == b & p:
                                match_and_break = True
                                break
                        if match_and_break:
                            continue
                        r = [v for k,v in dic.items() if k & b == b]
                        if len(r) == 0:
                            continue
                        cnt_all = len([f for f in r])
                        cnt_unmatch = len([f for f in r if f == 1])
                        if cnt_unmatch/cnt_all > error_tolerance:
                            continue

                        raw_perf_n.append([ii for ii in p_list[i]])
                        raw_perf2_n.append(b)       
#                 print("raw_perf_n", raw_perf_n)
            else:
                for s in range(max_dnf_len):
                    len_dnf = s + 1

                    l = [ii for ii in range(numvars)]
                    p_list = list(itertools.combinations(l, len_dnf))
                    print(str(len_dnf) + " variable patterns")
                    if len(p_list) > 1000000:
                        print("Skipping because " + str(len(p_list)) + " combinations is too many")
                        break
                    true_test_pass = True
                    for i in range(len(p_list)):
                        match_and_break = False
                        b = DNF_Regression_solver.convTuple2bin(p_list[i], numvars)
                        for p in raw_perf2_n:
                            if p == b & p:
                                match_and_break = True
                                break
                        if match_and_break:
                            continue
                        r = [v for k,v in dic_opp.items() if k & b == b]
                        if len(r) == 0:
                            continue
                        cnt_all = len([f for f in r])
                        cnt_unmatch = len([f for f in r if f == 0])
                        if cnt_unmatch/cnt_all > error_tolerance:
                            continue

                        raw_perf_n.append([ii for ii in p_list[i]])
                        raw_perf2_n.append(b)  
        
        for dn in raw_perf_n:
            dnf_perf_n.append(sorted(list(set([inp[0][ii] for ii in dn]))))

        
        print("size of false dnf in negative form " + str(len(dnf_perf_n)))
        if len(dnf_perf_n) > 10000:
            print("cutting to only 10000 of false dnf in negative form randomly as they are too many")
            dnf_perf_n = random.sample(dnf_perf_n, 10000)
        
        set_dnf_true = set(["(" + s + ")" for s in [" & ".join(a) for a in dnf_perf]])
        dnf_common = None
        set_dnf_false = None
        if check_false:
            cnf = None
            if check_negative:
                cnf = "(" + ") & (".join([" | ".join(a) for a in [[a[2:] if a[0:2] == "n_" else "n_" + a for a in aa] for aa in dnf_perf_n]]) + ")"
                cnf_list = [(a[2:] if a[0:2] == "n_" else "n_" + a for a in aa) for aa in dnf_perf_n]
            else:
                cnf = "(" + ") & (".join([" | ".join(a) for a in [[a for a in aa] for aa in dnf_perf_n]]) + ")"
                cnf_list = [(a for a in aa) for aa in dnf_perf_n]
#             cnf = str(boolalg.to_cnf(cnf, simplify=True, force=True))
#             set_dnf_false = set([word.strip() for word in str(boolalg.to_dnf(cnf, simplify=True, force=True)).split("|")])
            print("Converting from CNF to DNF for false expressions..")
            my_dnf = DNF_Regression_solver.cnf_to_dnf(cnf_list)
            print("Simplifying CNF for false expressions..")
            set_dnf_false = set(["(" + " & ".join(a) + ")" for a in [[a for a in aa] for aa in my_dnf]])
            dnf_common = set_dnf_true & set_dnf_false

        else:
            dnf_common = set_dnf_true
        
        self.expression = " | ".join(dnf_common)
        
        self.expression_common = " | ".join(dnf_common)
        self.expression_true = " | ".join(set_dnf_true)
        if self.expression == "":
            self.expression = self.expression_true
        self.expression_false = " | ".join(set_dnf_false)
        self.expression_union = " | ".join(set_dnf_true | set_dnf_false)
        
            
        if check_false:
            print("")
            print("DNF COMMON - " + str(len(dnf_common)))
            print("--------------------------------")

            if len(dnf_common) > 0:
                print(" | ".join(dnf_common))
            
            
        print("")
        print("DNF TRUE - " + str(len(set_dnf_true)))
        print("--------------------------------")

        if len(set_dnf_true) > 0:
            print(" | ".join(set_dnf_true))

        if check_false:
            print("")
            print("DNF FALSE - " + str(len(set_dnf_false)))
            print("--------------------------------")
            if len(set_dnf_false) > 0:
                print(" | ".join(set_dnf_false))
            
        perm_vars = list(set([xx for x in dnf_perf for xx in x] + [xx for x in dnf_perf_n for xx in x]))
        not_picked = [inp[0][ii] for ii in range(len(inp[0])-1) if inp[0][ii] not in perm_vars]

        print("")
        print("Unsolved variables - " + str(len(not_picked)) + "/" + str(len(inp[0])-1))
        print("--------------------------------")
        print(not_picked)
        print("")
        
        return inp






###### SAMPLE EXECUTION #########

import copy
from sklearn.metrics import precision_score, recall_score, f1_score

file_path = '/kaggle/input/tomio1/dnf_regression.txt'

reg = DNF_Regression_solver()
inp = reg.train(file_path=file_path, error_tolerance=0.03, check_negative=True)

answer = [int(inp[i][-1]) for i in range(1, len(inp), 1)]

inp = [row[:-1] for row in inp]
  
# print(inp)
res = reg.solve(inp, used_expression="false")
# res = reg.solve(inp)
print("Predicted")
print(res)
print("Actual")
print(answer)

precision = precision_score(answer, res)
recall = recall_score(answer, res)
f1 = f1_score(answer, res)

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

