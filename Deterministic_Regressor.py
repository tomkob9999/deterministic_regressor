# Name: Deterministic_Regressor
# Author: tomio kobayashi
# Version: 2.9.2
# Date: 2024/01/17

import itertools
from sympy.logic import boolalg
import numpy as np

import sklearn.datasets
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sympy import simplify

class Deterministic_Regressor:
# This version has no good matches
# Instead, all true matches are first added, and some are removed when 
# false unmatch exists and there is no corresponding other rule
    def __init__(self):
        self.expression_true = ""
        self.expression_false = ""
        self.true_confidence = {}
        self.false_confidence = {}
        
        self.tokens = []
        self.dic_segments = {}
        
        self.last_solve_expression = ""
        
        self.check_negative=False
        
        self.expression_opt = ""
        self.by_two = -1
        self.opt_f1 = 0.001
        
        self.children = []
        
        self.whole_rows = []
        self.test_rows = []
        self.train_rows = []
        
        

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
    
    
    def simplify_dnf(s, use_cnf=False):
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
        filtered_lists = [sorted(list(f)) for f in sorted(filtered_sets)]
        filtered_lists = [(" " + tok2 + " ").join(f) for f in sorted(filtered_lists)]
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
            
        
#     def solve(self, inp_p, check_negative=True, use_expression="union", confidence_thresh=3, power_level=None):
    def solve(self, inp_p, use_expression="union", confidence_thresh=3, power_level=None):
        
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp_p[i][j]) for j in range(len(inp_p[i]))] for i in range(len(inp_p))]
        
        max_power = 64
        
        true_confidence_thresh = 0
        false_confidence_thresh = 0
        if power_level is not None:
            if power_level > max_power or power_level < 0:
                print("power_level must be between 0 and 32")
                return

            if len(self.true_confidence) == 0:
                    true_confidence_thresh = 0  
            else:
                max_freq = max([v for k, v in self.true_confidence.items()])
                min_freq = min([v for k, v in self.true_confidence.items()])
    #             print("max_freq true", max_freq)
    #             print("min_freq true", min_freq)
                this_power = int(power_level/max_power * (max_freq-min_freq) + 0.9999999999)

                if max_freq - this_power < 0:
                    true_confidence_thresh = 0
                else:
                    true_confidence_thresh = max_freq - this_power

            if len(self.false_confidence) == 0:
                    false_confidence_thresh = 0  
            else:
                max_freq = max([v for k, v in self.false_confidence.items()])
                min_freq = min([v for k, v in self.false_confidence.items()])
    #             print("max_freq false", max_freq)
    #             print("min_freq false", min_freq)
                this_power = int(power_level/max_power * (max_freq-min_freq) + 0.9999999999)
                if max_freq - this_power < 0:
                    false_confidence_thresh = 0
                else:
                    false_confidence_thresh = max_freq - this_power
        else:
            true_confidence_thresh = confidence_thresh
            false_confidence_thresh = confidence_thresh
        
        print("true_confidence_thresh:", true_confidence_thresh)
        print("false_confidence_thresh:", false_confidence_thresh)
        print("Input Records:", len(inp)-1)
        
        numvars = len(inp[0])

        if self.check_negative:
            for i in range(numvars):
                inp[0].insert(i+numvars, "n_" + inp[0][i])
            for j in range(1, len(inp), 1):
                for i in range(numvars):
                    inp[j].insert(i+numvars,0 if inp[j][i] == 1 else 1)
            numvars *= 2

        tokens = inp[0]
        inp_list = [row for row in inp[1:]]
        
        res = list(range(len(inp_list)))
        expr = ""
        
        true_exp = self.expression_true
        false_exp = self.expression_false
        active_true_clauses = 0
        active_false_clauses = 0
        if confidence_thresh > 0:
            true_list = []
            for s in true_exp.split("|"):
                s = s.strip()
                if s in self.true_confidence:
                    if self.true_confidence[s] >= true_confidence_thresh:
                        true_list.append(s)
                        active_true_clauses += 1
                else:
                    true_list.append(s)
                    active_true_clauses += 1
            true_exp = " | ".join(true_list)
            
            false_list = []
            for s in false_exp.split("&"):
                s = s.strip()
                if s in self.false_confidence:
                    if self.false_confidence[s] >= false_confidence_thresh:
                        false_list.append(s)
                        active_false_clauses += 1
                else:
                    false_list.append(s)
                    active_false_clauses += 1
            false_exp = " & ".join(false_list)
        else:
            active_true_clauses = len(true_exp.split("|"))
            active_false_clauses = len(false_exp.split("&"))
            
        if use_expression == "true":
            if true_exp == "":
                print("The true expression is not available")
                return []
            expr = true_exp
            print(str(active_true_clauses) + " true clauses activated")
        elif use_expression == "false":
            if false_exp == "":
                print("The false expression is not available")
                return []
            expr = false_exp
            print(str(active_false_clauses) + " false clauses activated")
        elif use_expression == "common":
            if true_exp == "":
                print("The true expression is not available")
                return []
            if false_exp == "":
                print("The false expression is not available")
                return []
            expr = "(" + true_exp + ") & (" + false_exp + ")"
            print(str(active_true_clauses) + " true clauses activated")
            print(str(active_false_clauses) + " false clauses activated")
        else: # union case
            if true_exp == "":
                print("The true expression is not available")
                return []
            if false_exp == "":
                print("The false expression is not available")
                return []
            expr = "(" + true_exp + ") | (" + false_exp + ")"
            print(str(active_true_clauses) + " true clauses activated")
            print(str(active_false_clauses) + " false clauses activated")


        print("Solver Expression:")
        
        print(self.replaceSegName(expr).replace("(n_", "(NOT ").replace(" n_", " NOT "))
        
        self.last_solve_expression = expr

        for i in range(len(inp_list)):
            res[i] = Deterministic_Regressor.myeval(inp_list[i], tokens, expr)
            
        if res is None:
            return []
        
        return res
    
        
    def solve_direct(self, inp_p, expression):
        
        self.last_solve_expression = expression
           
        inp = copy.deepcopy(inp_p)
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]
        # SHOULD NOT DISCRETIXED DURING SOLVE
#         inp = self.discretize_data(inp, self.by_two, silent=True)

        numvars = len(inp[0])

        if self.check_negative:
            for i in range(numvars):
                inp[0].insert(i+numvars, "n_" + inp[0][i])
            for j in range(1, len(inp), 1):
                for i in range(numvars):
                    inp[j].insert(i+numvars,0 if inp[j][i] == 1 else 1)
            numvars *= 2

#         print("inp[0]", inp[0])
        
        tokens = inp[0]
        inp_list = [row for row in inp[1:]]
        
        res = list(range(len(inp_list)))
        
#         tokens = inp[0]
#         print("tokens", tokens)
#         inp_list = [row for row in inp[1:]]
#         res = list(range(len(inp_list)))
        
        
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

        return sampled_rows

#     #     WORKS ONLY FOR TOY DATA, NO USE USUALLY BUT KEEP FOR REFERENCE
#     def train_simple(self, file_path=None, data_list=None, max_dnf_len=4, by_two=1):

#         # Example usage:
#         # file_path = '/kaggle/input/dnf-regression/dnf_regression.txt'
#         # file_path = '/kaggle/input/tomio5/dnf_regression.txt'
        
#         inp = None
#         if file_path is not None:
#             with open(file_path, 'r') as f:
#                 inp = [line.strip().split('\t') for line in f]
#         else:
#             inp = data_list

        
#         print("Train Records:", len(inp)-1)
    
#         inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]
        
#         print("Discretizing...")
#         inp = self.discretize_data(inp, by_two)
#         print("")
        
#         imp_before_row_reduction = copy.deepcopy(inp)
# # # # # ############## COMMENT OUT UNLESS TESTING ############## 
# #         CUT_PCT = 60
# #         print("NUM RECS BEFORE REDUCTION FOR TEST", len(inp))
# #         inp = Deterministic_Regressor.reduce_rows_except_first(inp, CUT_PCT)
# #         print("NUM RECS AFTER REDUCTION FOR TEST", len(inp))
# # # # # ############## COMMENT OUT UNLESS TESTING ############## 

        
#         numvars = len(inp[1])-1

#         print("Columns:")
#         print(inp[0])
#         self.tokens = copy.deepcopy(inp[0])
        
#         print("")
            
#         dic = dict()
                    
#         dic_opp = dict()
        
#         true_set = set()
#         false_set = set()
        
#         for i in range(1, len(inp), 1):
#             s = ""
#             cnt_1 = 0
#             cnt_0 = 0
#             for j in range(len(inp[i]) - 1):
#                 s += str(inp[i][j])
#                 if inp[i][j] == 1:
#                     cnt_1 += 1
#                 else:
#                     cnt_0 += 1
                    
#             truefalse = inp[i][len(inp[i]) - 1]
#             dic[int(s, 2)] = truefalse
#             if truefalse == 1:
#                 if cnt_1 <= max_dnf_len:
#                     true_set.add(s)
#             else:
#                 if cnt_0 <= max_dnf_len:
#                     false_set.add(s)
                    
#         true_dnf = Deterministic_Regressor.simplify_dnf("(" + ") | (".join([" & ".join([self.tokens[i] for i in range(len(f)) if f[i] == "1"]) for f in true_set]) + ")")
#         false_cnf = Deterministic_Regressor.simplify_dnf("(" + ") & (".join([" | ".join([self.tokens[i] for i in range(len(f)) if f[i] == "0"]) for f in false_set]) + ")", use_cnf=True)
#         if true_dnf == "()":
#             true_dnf = ""
#         if false_cnf == "()":
#             false_cnf = ""
#         self.expression_true = true_dnf
#         self.expression_false = false_cnf
            
#         print("")
#         print("TRUE DNF - " + str(len(true_dnf.split("|"))))
#         print("--------------------------------")

#         if len(true_dnf) > 0:
#             print(self.replaceSegName(self.expression_true))
            

#         print("")
#         print("FALSE CNF - " + str(len(false_cnf.split("&"))))
#         print("--------------------------------")
#         if len(false_cnf) > 0:
#             print(self.replaceSegName(self.expression_false))
            
#         return imp_before_row_reduction
    
    def clean_and_discretize(self, inp, by_two):
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]
#         return self.discretize_data(inp, by_two)
        matrix = self.discretize_data(inp, by_two)
        head = matrix[0]
        return [head] + [[int(mm) for mm in m] for m in matrix[1:]]
        
    def train(self, file_path=None, data_list=None, max_dnf_len=4, check_false=True, check_negative=False, error_tolerance=0.02, min_match=0.03, use_approx_dnf=False, redundant_thresh=1.00):

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
        
        print("Train Records:", len(inp)-1)
    
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]
        
#         print("Discretizing...")
#         inp = self.discretize_data(inp, by_two)
#         print("")
        for r in inp[1:]:
            for c in r:
                if c != 1 and c != 0:
                    print("The data contents needs to be 1 or 0", c)
                    return []
        
        imp_before_row_reduction = copy.deepcopy(inp)
# # # ############## COMMENT OUT UNLESS TESTING ############## 
#         CUT_PCT = 90
#         print("NUM RECS BEFORE REDUCTION FOR TEST", len(inp))
#         inp = Deterministic_Regressor.reduce_rows_except_first(inp, CUT_PCT)
#         print("NUM RECS AFTER REDUCTION FOR TEST", len(inp))
# # ############## COMMENT OUT UNLESS TESTING ############## 

        self.check_negative = check_negative
        
        numvars = len(inp[1])-1
        
        if check_negative:
            for i in range(numvars):
                inp[0].insert(i+numvars, "n_" + inp[0][i])
            for j in range(1, len(inp), 1):
                for i in range(numvars):
                    inp[j].insert(i+numvars, 0 if inp[j][i] == 1 else 1)
            numvars *= 2

        print("Columns:")
        print(inp[0])
        self.tokens = copy.deepcopy(inp[0])
        print("")
        
        print("Redundant Pairs:")
        numrows = len(inp)-1
        redundant_cols = set()
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
        
        dic = dict()
                    
        dic_opp = dict()
        
        true_list = []
        false_list = []
        for i in range(1, len(inp), 1):
            s = ""
            for j in range(len(inp[i]) - 1):
#                 s += str(inp[i][j])
                s += str(int(inp[i][j]))
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
                s += str(inp_oppo[i][j])
            truefalse = inp_oppo[i][len(inp_oppo[i]) - 1]
            dic_opp[int(s, 2)] = truefalse

                
        MAX_REPS = 1500000
        
        print("Deriving true expressions...")
        dnf_perf = list()
        raw_perf = list()
        raw_perf2 = list()

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
                    continue
#                 if cnt_all - cnt_unmatch < min_match:
                if (cnt_all - cnt_unmatch)/numrows < min_match:
                    continue

                raw_perf.append([ii for ii in p_list[i]])
                raw_perf2.append(b)

                self.true_confidence["(" + " & ".join(sorted(list(set([inp[0][ii] for ii in p_list[i]])))) + ")"] = cnt_all - cnt_unmatch
        
        
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
                    
                    p_list = [p for p in p_list if not any([pp in redundant_cols for pp in p])]
            
                    print(str(len_dnf) + " variable patterns")
                    if len(p_list) > MAX_REPS:
                        print("Skipping because " + str(len(p_list)) + " combinations is too many")
                        break

                    true_test_pass = True
                    for i in range(len(p_list)):
                        match_and_break = False
                        b = Deterministic_Regressor.convTuple2bin(p_list[i], numvars)
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

#                         if cnt_all - cnt_unmatch < min_match:
                        if (cnt_all - cnt_unmatch)/numrows < min_match:
                            continue
                            
                        raw_perf_n.append([ii for ii in p_list[i]])
                        raw_perf2_n.append(b)       
                        self.false_confidence["(" + " | ".join(sorted(list(set([inp[0][ii][2:] if inp[0][ii][0:2] == "n_" else "n_" + inp[0][ii] for ii in p_list[i]])))) + ")"] = cnt_all - cnt_unmatch
            else:
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

#                         if cnt_all - cnt_unmatch < min_match:
                        if (cnt_all - cnt_unmatch)/numrows < min_match:
                            continue
                            
                        raw_perf_n.append([ii for ii in p_list[i]])
                        raw_perf2_n.append(b)  
                        self.false_confidence["(" + " | ".join(sorted(list(set([inp[0][ii] for ii in p_list[i]])))) + ")"] = cnt_all - cnt_unmatch
   
        for dn in raw_perf_n:
            dnf_perf_n.append(sorted(list(set([inp[0][ii] for ii in dn]))))
            
        print("size of false cnf " + str(len(dnf_perf_n)))
        
        set_dnf_true = set(["(" + s + ")" for s in [" & ".join(a) for a in dnf_perf]])
        dnf_common = None
        set_dnf_false = None
        if check_false:
            cnf = None
            cnf_list = None
            if check_negative:
                if len(dnf_perf_n) > 0:
                    cnf = "(" + ") & (".join([" | ".join(a) for a in [[a[2:] if a[0:2] == "n_" else "n_" + a for a in aa] for aa in dnf_perf_n]]) + ")"
                else:
                    cnf = ""
                set_cnf_false = [[a[2:] if a[0:2] == "n_" else "n_" + a for a in aa] for aa in dnf_perf_n]
            else:
                if len(dnf_perf_n) > 0:
                    cnf = "(" + ") & (".join([" | ".join(a) for a in [[a for a in aa] for aa in dnf_perf_n]]) + ")"
                else:
                    cnf = ""
                set_cnf_false = [[a for a in aa] for aa in dnf_perf_n]

        self.expression_true = " | ".join(set_dnf_true)
        self.expression_true = Deterministic_Regressor.simplify_dnf(self.expression_true)
        self.expression_false = Deterministic_Regressor.simplify_dnf(cnf, use_cnf=True)
        

            
        print("")
        print("TRUE DNF - " + str(len(set_dnf_true)))
        print("--------------------------------")

        if len(set_dnf_true) > 0:
            print(self.replaceSegName(self.expression_true).replace("(n_", "(NOT ").replace(" n_", " NOT "))
            

        if check_false:
            print("")
            print("FALSE CNF - " + str(len(set_cnf_false)))
            print("--------------------------------")
            if len(set_cnf_false) > 0:
                print(self.replaceSegName(self.expression_false).replace("(n_", "(NOT ").replace(" n_", " NOT "))
            
        perm_vars = list(set([xx for x in dnf_perf for xx in x] + [xx for x in dnf_perf_n for xx in x]))
        
        not_picked = [self.replaceSegName(inp[0][ii]) if self.replaceSegName(inp[0][ii])[:2] != "n_" else "NOT " + self.replaceSegName(inp[0][ii])[2:] for ii in range(len(inp[0])-1) if inp[0][ii] not in perm_vars]

        print("")
        print("Unsolved variables - " + str(len(not_picked)) + "/" + str(len(inp[0])-1))
        print("--------------------------------")
        print(not_picked)
        print("")
        
        return imp_before_row_reduction


    def optimize_params(self, test_data, answer, elements_count_penalty=1.0, solve_method=["common", "union"]):
        
        inp = test_data
        
        best_ee_sofar = 0
        ct_now = 0

        MAX_POWER_LEVEL = 64
        jump = int(MAX_POWER_LEVEL/2)
        ct_opt = 0
        expr_opt = ""
        win_option_sofar = ""
        opt_precision_sofar = 0
        opt_recall_sofar = 0
        opt_f1_sofar = 0

#         ops = ["union", "common", "true", "false"]
#         if len(used_options) > 0:
#             ops = used_options
        ops = solve_method
        
        
        doexit = False
        for i in range(100):
            print("")
            print("")
            print("##### Power Level", ct_now, "######")
        
            best_ee = 0
            win_option = ""
            win_expr = ""
            opt_precision = 0
            opt_recall = 0
            opt_f1 = 0
            for o in ops:
                print("")
                print("******* " + o + " ********")
                res = self.solve(inp, use_expression=o, power_level=ct_now)

                if len(res) > 0:

                    precision = precision_score(answer, res)
                    recall = recall_score(answer, res)
                    f1 = f1_score(answer, res)
                    print(f"Precision: {precision * 100:.2f}%")
                    print(f"Recall: {recall * 100:.2f}%")
                    print(f"F1 Score: {f1 * 100:.2f}%")
                    ee = (f1 +min(precision,recall))/2-(len(self.last_solve_expression.split("&"))+len(self.last_solve_expression.split("|")))/3000*elements_count_penalty
                    print(f"Effectiveness & Efficiency Score: {ee * 100:.3f}%")
                    if best_ee < ee:
                        best_ee = ee
                        win_option = o
                        win_expr = self.last_solve_expression
                        opt_precision = precision
                        opt_recall = recall
                        opt_f1 = f1
            if best_ee_sofar < best_ee:
                ct_opt = ct_now
                best_ee_sofar = best_ee
                ct_now = ct_now + jump
                win_option_sofar = win_option
                expr_opt = win_expr
                opt_precision_sofar = opt_precision
                opt_recall_sofar = opt_recall
                opt_f1_sofar = opt_f1
                if ct_now > MAX_POWER_LEVEL:
                    doexit = True
            elif jump == 1:
                doexit = True
            elif ct_now == 0:
                print("#################################")
                print("")
                print("SORRY NO SOLUTION FOUND")
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
                print("OPTIMUM POWER LEVEL is", ct_opt, "with", win_option_sofar)
                print("")
                print(f"Precision: {opt_precision_sofar * 100:.2f}%")
                print(f"Recall: {opt_recall_sofar * 100:.2f}%")
                print(f"F1 Score: {opt_f1_sofar * 100:.2f}%")
                print(f"Effectiveness & Efficiency Score: {best_ee_sofar * 100:.3f}%")
                print("Expression:")
                print(self.replaceSegName(expr_opt).replace("(n_", "(NOT ").replace(" n_", " NOT "))
                print("")
                print("#################################")
                print("")
        
                self.expression_opt = expr_opt
                self.opt_f1 = f1
                
                return win_option_sofar, ct_opt

    def optimize_compact(self, test_data, answer, cnt_out=8):

        inp = test_data
        
        print("false CNF analysis started")
        false_clauses = sorted([(v, k) for k, v in self.false_confidence.items()], reverse=True)
        
        expr = ""
        false_recall = {}
        false_best_expr = ""
        min_fp = 9999999
        min_fn = 9999999
        
        best_ee = 0
        if len(false_clauses) > 0:

            res = self.solve_direct(inp, false_clauses[0][1])
    
            precision = precision_score(answer, res)
            recall = recall_score(answer, res)
            f1 = f1_score(answer, res)
            best_ee = (f1 + min(precision,recall))/2
            
                
#             print("optimize_max 2")

            conf_matrix = confusion_matrix(answer, res)
            tn, fp, fn, tp = conf_matrix.ravel()
            min_fp = fp
            min_fn = fn

            false_best_expr = false_clauses[0][1]
            
            cnt = 0
            for i in range(1, len(false_clauses), 1):
                cnt = cnt + 1
                if cnt_out < cnt:
                    break
                    
                if i % 10 == 0:
                    print(str(i) + "/" + str(len(false_clauses)) + " completed" )

                expr = false_best_expr + " & " + false_clauses[i][1]
                res = self.solve_direct(inp, false_clauses[i][1])
                conf_matrix = confusion_matrix(answer, res)
                tn, fp, fn, tp = conf_matrix.ravel()
                precision = precision_score(answer, res)
                recall = recall_score(answer, res)
#                 print("precision", precision)
#                 print("recall", recall)
                f1 = f1_score(answer, res)
                ee = (f1 + min(precision,recall))/2
                ee = precision
                if best_ee < ee:
#                 if min_fp > fp and (0.2 > fn/len(answer)):
#                 if min_fp + min_fn > fp + fn:
                    best_ee = ee

                    min_fp = fp
                    min_fn = fn
                    false_best_expr = expr
                    cnt = 0

                    if false_best_expr == "":
                        false_best_expr = "(" + false_clauses[i][1] + ")"
                    else:
                        false_best_expr = false_best_expr + " | (" + false_clauses[i][1] + ")"
#         best_ee = 0   
#         print("false_best_expr", false_best_expr)
        
        print("true DNF analysis started")
        true_clauses = sorted([(v, k) for k, v in self.true_confidence.items()], reverse=True)
        true_best_expr = ""
        cnt = 0
        for i in range(1, len(true_clauses), 1):
            cnt = cnt + 1
            if cnt_out < cnt:
                break
                    
            if i % 10 == 0:
                print(str(i) + "/" + str(len(true_clauses)) + " completed" )
            
            if true_best_expr == "":
                if false_best_expr == "":
                    expr = "(" + true_clauses[i][1] + ")"
                else:
                    expr = false_best_expr + " | (" + true_clauses[i][1] + ")"
            else:
                if false_best_expr == "":
                    expr = "(" + true_best_expr + " | " + true_clauses[i][1] + ")"
                else:
                    expr = false_best_expr + " | (" + true_best_expr + " | " + true_clauses[i][1] + ")"
            
#             print("try true expr", expr)
            res = self.solve_direct(inp, expr)
    
            conf_matrix = confusion_matrix(answer, res)
            tn, fp, fn, tp = conf_matrix.ravel()
            
            precision = precision_score(answer, res)
            recall = recall_score(answer, res)
            f1 = f1_score(answer, res)
            ee = (f1 + min(precision,recall))/2
#             print("best_ee", best_ee)
#             print("ee", ee)
            if best_ee < ee:
                best_ee = ee
                min_fp = fp
                min_fn = fn
                if true_best_expr == "":
                    true_best_expr = true_clauses[i][1]
                else:
                    true_best_expr = true_best_expr + " | " + true_clauses[i][1]
                cnt = 0

#         print("true_best_expr", true_best_expr)
        
        final_expr = ""
        if true_best_expr == "" and false_best_expr == "":
            final_expr = ""
        if true_best_expr != "" and false_best_expr == "":
            final_expr = true_best_expr
        if true_best_expr == "" and false_best_expr != "":
            final_expr = false_best_expr
        if true_best_expr != "" and false_best_expr != "":
            final_expr = false_best_expr + " | (" + true_best_expr + ")"

        if final_expr != "":
            
            print("assessment of the optimal solution")
            res = self.solve_direct(inp, final_expr)

            if len(res) > 0:
                print("")
                print("#### DERIVED OPTIMUM EXPRESSION ####")
                print("")
                precision = precision_score(answer, res)
                recall = recall_score(answer, res)
                f1 = f1_score(answer, res)
                print(f"Precision: {precision * 100:.2f}%")
                print(f"Recall: {recall * 100:.2f}%")
                print(f"F1 Score: {f1 * 100:.2f}%")

                self.opt_f1 = f1
                
                print(self.replaceSegName(final_expr).replace("(n_", "(NOT ").replace(" n_", " NOT "))
                
                print("")
                
                self.expression_opt = final_expr
                
                return final_expr
            
            
    def random_split_matrix(matrix, divide_by=2):
        rows = list(matrix)  # Convert to list for easier shuffling
        random.shuffle(rows)  # Shuffle rows in-place
        split_index = len(rows) // divide_by  # Integer division for equal or near-equal halves
        return rows[:split_index], rows[split_index:]

    def train_and_optimize(self, data_list=None, max_dnf_len=4, check_false=True, check_negative=False, error_tolerance=0.02, 
                       min_match=0.03, use_approx_dnf=False, redundant_thresh=1.00, solve_method=["common", "union"], elements_count_penalty=1.0, 
                           use_compact_opt=False, cnt_out=8):
        
        print("Training started...")
        
#         self.by_two = by_two
        
        headers = data_list[0]
        data_list2 = data_list[1:]
        
        train_data, valid_data = Deterministic_Regressor.random_split_matrix(data_list2)

        train_inp = [headers] + train_data
        
        self.train(data_list=train_inp, max_dnf_len=max_dnf_len, check_false=check_false, check_negative=check_negative, 
                        error_tolerance=error_tolerance, min_match=min_match, use_approx_dnf=use_approx_dnf, redundant_thresh=redundant_thresh)

        print("Optimization started...")
        inp = [headers] + valid_data
           
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]
#         inp = self.discretize_data(inp, by_two, silent=True)
    
        if check_negative:
            for i in range(numvars):
                inp[0].insert(i+numvars, "n_" + inp[0][i])
            for j in range(1, len(inp), 1):
                for i in range(numvars):
                    inp[j].insert(i+numvars, 0 if inp[j][i] == 1 else 1)
            numvars *= 2
                
        answer = [int(inp[i][-1]) for i in range(1, len(inp), 1)]
        inp = [row[:-1] for row in inp]
             
        if use_compact_opt:
            return self.optimize_compact(inp, answer, cnt_out=cnt_out)
        else:
            return self.optimize_params(inp, answer, solve_method=solve_method, elements_count_penalty=1.0)
    
    def train_and_optimize_bulk(self, data_list, expected_answers, max_dnf_len=4, check_false=True, check_negative=False, error_tolerance=0.02,  
                   min_match=0.03, use_approx_dnf=False, redundant_thresh=1.00, solve_method=["common", "union"], elements_count_penalty=1.0, use_compact_opt=False, cnt_out=8):

        self.children = [Deterministic_Regressor() for _ in range(len(expected_answers))]

        for i in range(len(self.children)):
            print("Child", i)
            
            self.children[i].dic_segments = copy.deepcopy(self.dic_segments)
            d_list = copy.deepcopy(data_list)
            d_list[0].append("res")
            
            for k in range(len(d_list)-1):
                d_list[k+1].append(expected_answers[i][k])
            self.children[i].train_and_optimize(data_list=d_list, max_dnf_len=max_dnf_len, check_false=check_false, check_negative=check_negative, error_tolerance=error_tolerance, 
                    min_match=min_match, use_approx_dnf=use_approx_dnf, redundant_thresh=redundant_thresh, solve_method=solve_method, 
                                                elements_count_penalty=elements_count_penalty, use_compact_opt=use_compact_opt, cnt_out=cnt_out)

            
            
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
                    new_res[i] = numbers[k]
                    break
                if k == len(numbers) - 1:
                    new_res[i] = numbers[k]
                    

        return new_res
    
    def train_and_optimize_class(self, data_list, expected_answers, max_dnf_len=4, check_false=True, check_negative=False, error_tolerance=0.02, 
               min_match=0.03, use_approx_dnf=False, redundant_thresh=1.00, solve_method=["common", "union"], elements_count_penalty=1.0, use_compact_opt=False, cnt_out=10):
        
        answers = [[0 for _ in range(len(expected_answers))] for _ in range(max(expected_answers)+1)]
        for i in range(len(answers[0])):
            answers[expected_answers[i]][i] = 1
        self.train_and_optimize_bulk(data_list=data_list, expected_answers=answers, max_dnf_len=max_dnf_len, check_false=check_false, 
                    check_negative=check_negative, error_tolerance=error_tolerance, 
                    min_match=min_match, use_approx_dnf=use_approx_dnf, redundant_thresh=redundant_thresh, solve_method=solve_method, 
                                     elements_count_penalty=elements_count_penalty, use_compact_opt=use_compact_opt, cnt_out=cnt_out)
    
    def prepropcess(self, whole_rows, by_two, splitter=3):
        self.whole_rows = self.clean_and_discretize(whole_rows, by_two)
        headers = self.whole_rows
        data = self.whole_rows[1:]
        random.shuffle(data)  # Shuffle rows in-place
        split_index = len(data) // splitter  # Integer division for equal or near-equal halves
        self.test_rows = data[:split_index]
        print("len(self.test_rows)", len(self.test_rows))
        self.train_rows = data[split_index:]
        print("len(self.train_rows)", len(self.train_rows))
    
    def get_train_dat_wo_head(self):
        return [row[:-1] for row in self.train_rows]
    def get_train_res_wo_head(self):
        return [row[-1] for row in self.train_rows]
    def get_train_dat_with_head(self):
        return [self.whole_rows[0][:-1]] + [row[:-1] for row in self.train_rows]
#     def get_train_res_with_head(self):
#         return [self.whole_rows[0][:-1]] + [row[-1] for row in self.train_rows]
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
#     def get_test_res_with_head(self):
#         return [self.whole_rows[0][:-1]] + [row[-1] for row in self.test_rows]
    def get_test_datres_wo_head(self):
        return self.test_rows[1:]
    def get_test_datres_with_head(self):
        return [self.whole_rows[0]] + self.test_rows
    

    def show_stats(predicted, actual, average="binary", elements_count_penalty=1.0):
        
        answer = actual
        res = predicted
        
        precision = precision_score(answer, res, average=average)
        recall = recall_score(answer, res, average=average)
        f1 = f1_score(answer, res, average=average)
        print("")
        print("####### PREDICTION STATS #######")
        print("")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")
        print("")
        print("##############")
        print("")

    