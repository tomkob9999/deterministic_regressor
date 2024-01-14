# Name: Deterministic_Regressor
# Author: tomio kobayashi
# Version: 2.6.2
# Date: 2024/01/14

import itertools
from sympy.logic import boolalg
import numpy as np

import sklearn.datasets
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
        
#     def cnf_to_dnf(cnf):
#         dnf_clauses = []

#         for clause in cnf:
#             dnf_clause = []
#             for literal in clause:
#                 if isinstance(literal, tuple):  # Handle negation
#                     dnf_clause.append((literal[0], not literal[1]))
#                 else:
#                     dnf_clause.append(literal)
#             dnf_clauses.append(dnf_clause)

#         dnf_result = []
#         if len(dnf_clauses) > 0:
#             for i in range(len(dnf_clauses[0])):
#                 dnf_result.append(sorted(list(set([dnf_clauses[j][i] for j in range(len(dnf_clauses))]))))

#         return dnf_result
    
    def remove_supersets(sets_list):
        result = []
        for s in sets_list:
            if not any(s != existing_set and s.issuperset(existing_set) for existing_set in sets_list):
                result.append(s)
        return result

#     def cnf_to_dnfx(cnf):
#         dnf = []
#         for clause in cnf:
#             dnf_clause = []
#             for literal in clause:
# #                 dnf_clause.append([literal])
#                 dnf_clause.append(literal)
#             dnf.append(dnf_clause)
#     #     return [list(x) for x in itertools.product(*dnf)]
#         dnfl = [list(x) for x in itertools.product(*dnf)]
# #         dnfl = [["".join(dd) for dd in d] for d in dnfl]
#         dnfl = [set(d) for d in dnfl]
#         filtered_sets = Deterministic_Regressor.remove_supersets(dnfl)
#         filtered_lists = [sorted(list(f)) for f in sorted(filtered_sets)]
# #         filtered_lists = [" & ".join(f) for f in sorted(filtered_lists)]
# #         str = "(" + ") | (".join(filtered_lists) + ")"
# #     #     print("str", str)
# #         return str
#         return filtered_lists


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
            
        
    def solve(self, inp_p, check_negative=True, use_expression="union", confidence_thresh=3, power_level=None):
        
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp_p[i][j]) for j in range(len(inp_p[i]))] for i in range(len(inp_p))]
        
        max_power = 64
        
        true_confidence_thresh = 0
        false_confidence_thresh = 0
        if power_level is not None:
            if power_level > max_power or power_level < 0:
                print("power_level must be between 0 and 32")
                return
            
            max_freq = max([v for k, v in self.true_confidence.items()])
            min_freq = min([v for k, v in self.true_confidence.items()])
            power_level = int(power_level/max_power * (max_freq-min_freq) + 0.9999999999)
            
            if max_freq - power_level < 0:
                true_confidence_thresh = 0
            else:
                true_confidence_thresh = max_freq - power_level
                
            max_freq = max([v for k, v in self.false_confidence.items()])
            min_freq = min([v for k, v in self.true_confidence.items()])
            power_level = int(power_level/max_power * (max_freq-min_freq) + 0.9999999999)
            if max_freq - power_level < 0:
                false_confidence_thresh = 0
            else:
                false_confidence_thresh = max_freq - power_level
            
        else:
            true_confidence_thresh = confidence_thresh
            false_confidence_thresh = confidence_thresh
        
        print("true_confidence_thresh:", true_confidence_thresh)
        print("false_confidence_thresh:", false_confidence_thresh)
        print("Input Records:", len(inp)-1)
        
        numvars = len(inp[0])

        if check_negative:
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
        print(self.replaceSegName(expr))
        
        self.last_solve_expression = expr
        
        for i in range(len(inp_list)):
            res[i] = Deterministic_Regressor.myeval(inp_list[i], tokens, expr)
        return res
    
    
#     def solve(self, inp_p, check_negative=True, use_expression="union", confidence_thresh=3, power_level=None):
#         inp = [[Deterministic_Regressor.try_convert_to_numeric(inp_p[i][j]) for j in range(len(inp_p[i]))] for i in range(len(inp_p))]
        
#         if power_level is not None:
#             max_freq = 0
#             if use_expression == "union" or use_expression == "common": 
#                 max_freq = min(max([v for k, v in self.true_confidence.items()]), max([v for k, v in self.false_confidence.items()]))
#             elif use_expression == "true":
#                 max_freq = max([v for k, v in self.true_confidence.items()])
#             else:
#                 max_freq = max([v for k, v in self.false_confidence.items()])
            
#             if max_freq - power_level < 0:
#                 confidence_thresh = 0
#             else:
#                 confidence_thresh = max_freq - power_level
# #         print("confidence_thresh:", confidence_thresh)
                
#         print("Input Records:", len(inp)-1)
        
#         numvars = len(inp[0])

#         if check_negative:
#             for i in range(numvars):
#                 inp[0].insert(i+numvars, "n_" + inp[0][i])
#             for j in range(1, len(inp), 1):
#                 for i in range(numvars):
#                     inp[j].insert(i+numvars,0 if inp[j][i] == 1 else 1)
#             numvars *= 2

#         tokens = inp[0]
#         inp_list = [row for row in inp[1:]]
        
#         res = list(range(len(inp_list)))
#         expr = ""
        
#         true_exp = self.expression_true
#         false_exp = self.expression_false
#         active_true_clauses = 0
#         active_false_clauses = 0
#         if confidence_thresh > 0:
#             true_list = []
#             for s in true_exp.split("|"):
#                 s = s.strip()
#                 if s in self.true_confidence:
#                     if self.true_confidence[s] >= confidence_thresh:
#                         true_list.append(s)
#                         active_true_clauses += 1
#                 else:
#                     true_list.append(s)
#                     active_true_clauses += 1
#             true_exp = " | ".join(true_list)
            
#             false_list = []
#             for s in false_exp.split("&"):
#                 s = s.strip()
#                 if s in self.false_confidence:
#                     if self.false_confidence[s] >= confidence_thresh:
#                         false_list.append(s)
#                         active_false_clauses += 1
#                 else:
#                     false_list.append(s)
#                     active_false_clauses += 1
#             false_exp = " & ".join(false_list)
#         else:
#             active_true_clauses = len(true_exp.split("|"))
#             active_false_clauses = len(false_exp.split("&"))
            
#         if use_expression == "true":
#             if true_exp == "":
#                 print("The true expression is not available")
#                 return []
#             expr = true_exp
#             print(str(active_true_clauses) + " true clauses activated")
#         elif use_expression == "false":
#             if false_exp == "":
#                 print("The false expression is not available")
#                 return []
#             expr = false_exp
#             print(str(active_false_clauses) + " false clauses activated")
#         elif use_expression == "common":
#             if true_exp == "":
#                 print("The true expression is not available")
#                 return []
#             if false_exp == "":
#                 print("The false expression is not available")
#                 return []
#             expr = "(" + true_exp + ") & (" + false_exp + ")"
#             print(str(active_true_clauses) + " true clauses activated")
#             print(str(active_false_clauses) + " false clauses activated")
#         else: # union case
#             if true_exp == "":
#                 print("The true expression is not available")
#                 return []
#             if false_exp == "":
#                 print("The false expression is not available")
#                 return []
#             expr = "(" + true_exp + ") | (" + false_exp + ")"
#             print(str(active_true_clauses) + " true clauses activated")
#             print(str(active_false_clauses) + " false clauses activated")


#         print("Solver Expression:")
#         print(self.replaceSegName(expr))
        
#         self.last_solve_expression = expr
        
#         for i in range(len(inp_list)):
#             res[i] = Deterministic_Regressor.myeval(inp_list[i], tokens, expr)
#         return res

    
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

    def discretize_data(self, data_list, by_four=1, silent=False):

        result_header = data_list[0][-1]
        result_values = [d[-1] for d in data_list[1:]]
        
        headers = data_list[0][:-1]
        values = [d[:-1] for d in data_list[1:]]
        data = pd.DataFrame(values, columns=headers) 

        
        cols = [c for c in data.columns]
        for c in cols:
            countNonBool = len(data[c]) - (data[c] == 0).sum() - (data[c] == 1).sum()
            if countNonBool > 0 and pd.api.types.is_numeric_dtype(data[c]):
                result_df = self.generate_segment_ranks(data, by_four*4, c, silent=silent)
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

    #     works only for toy data
    def train_simple(self, file_path=None, data_list=None, max_dnf_len=4, by_four=1):

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
        
        print("Discretizing...")
        inp = self.discretize_data(inp, by_four)
        print("")
        
        imp_before_row_reduction = copy.deepcopy(inp)
# # # ############## COMMENT OUT UNLESS TESTING ############## 
        CUT_PCT = 60
        print("NUM RECS BEFORE REDUCTION FOR TEST", len(inp))
        inp = Deterministic_Regressor.reduce_rows_except_first(inp, CUT_PCT)
        print("NUM RECS AFTER REDUCTION FOR TEST", len(inp))
# # # ############## COMMENT OUT UNLESS TESTING ############## 

        
        numvars = len(inp[1])-1

        print("Columns:")
        print(inp[0])
        self.tokens = copy.deepcopy(inp[0])
        
        print("")
            
        dic = dict()
                    
        dic_opp = dict()
        
        true_set = set()
        false_set = set()
        
        for i in range(1, len(inp), 1):
            s = ""
            cnt_1 = 0
            cnt_0 = 0
            for j in range(len(inp[i]) - 1):
                s += str(inp[i][j])
                if inp[i][j] == 1:
                    cnt_1 += 1
                else:
                    cnt_0 += 1
                    
            truefalse = inp[i][len(inp[i]) - 1]
            dic[int(s, 2)] = truefalse
            if truefalse == 1:
                if cnt_1 <= max_dnf_len:
                    true_set.add(s)
            else:
                if cnt_0 <= max_dnf_len:
                    false_set.add(s)
                    
        true_dnf = Deterministic_Regressor.simplify_dnf("(" + ") | (".join([" & ".join([self.tokens[i] for i in range(len(f)) if f[i] == "1"]) for f in true_set]) + ")")
        false_cnf = Deterministic_Regressor.simplify_dnf("(" + ") & (".join([" | ".join([self.tokens[i] for i in range(len(f)) if f[i] == "0"]) for f in false_set]) + ")", use_cnf=True)
        if true_dnf == "()":
            true_dnf = ""
        if false_cnf == "()":
            false_cnf = ""
        self.expression_true = true_dnf
        self.expression_false = false_cnf
            
        print("")
        print("TRUE DNF - " + str(len(true_dnf.split("|"))))
        print("--------------------------------")

        if len(true_dnf) > 0:
            print(self.replaceSegName(self.expression_true))
            

        print("")
        print("FALSE CNF - " + str(len(false_cnf.split("&"))))
        print("--------------------------------")
        if len(false_cnf) > 0:
            print(self.replaceSegName(self.expression_false))
            
        return imp_before_row_reduction
    
    
    def train(self, file_path=None, data_list=None, max_dnf_len=4, check_false=True, check_negative=False, error_tolerance=0.02, by_four=1, min_match=3, use_approx_dnf=False):

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
        
        print("Discretizing...")
        inp = self.discretize_data(inp, by_four)
        print("")
        
        imp_before_row_reduction = copy.deepcopy(inp)
# # # ############## COMMENT OUT UNLESS TESTING ############## 
#         CUT_PCT = 60
#         print("NUM RECS BEFORE REDUCTION FOR TEST", len(inp))
#         inp = Deterministic_Regressor.reduce_rows_except_first(inp, CUT_PCT)
#         print("NUM RECS AFTER REDUCTION FOR TEST", len(inp))
# # # ############## COMMENT OUT UNLESS TESTING ############## 

        
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
                if cnt_all - cnt_unmatch < min_match:
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

                        if cnt_all - cnt_unmatch < min_match:
                            continue
                            
                        raw_perf_n.append([ii for ii in p_list[i]])
                        raw_perf2_n.append(b)       
#                         self.false_confidence["(" + " | ".join(sorted(list(set([inp[0][ii] for ii in p_list[i]])))) + ")"] = cnt_all if cnt_unmatch == 0 else cnt_all/(cnt_unmatch+1)
                        self.false_confidence["(" + " | ".join(sorted(list(set([inp[0][ii] for ii in p_list[i]])))) + ")"] = cnt_all - cnt_unmatch
#                 print("raw_perf_n", raw_perf_n)
            else:
                for s in range(max_dnf_len):
                    len_dnf = s + 1

                    l = [ii for ii in range(numvars)]
                    p_list = list(itertools.combinations(l, len_dnf))
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

                        if cnt_all - cnt_unmatch < min_match:
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
        print("DNF TRUE - " + str(len(set_dnf_true)))
        print("--------------------------------")

        if len(set_dnf_true) > 0:
            print(self.replaceSegName(self.expression_true))
            

        if check_false:
            print("")
            print("CNF FALSE - " + str(len(set_cnf_false)))
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
        
#         return inp
        return imp_before_row_reduction


    def optimize_params(self, test_data, answer, elements_count_penalty=1.0, used_options=[]):

        inp = test_data
        
        best_ee_sofar = 0
        ct_now = 0

        jump = 32
        ct_opt = 0
        expr_opt = ""
        win_option_sofar = ""
        opt_precision_sofar = 0
        opt_recall_sofar = 0
        opt_f1_sofar = 0

        ops = ["union", "common", "true", "false"]
        if len(used_options) > 0:
            ops = used_options
        
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
                    ee = (f1 +min(precision,recall))/2-(len(reg.last_solve_expression.split("&"))+len(reg.last_solve_expression.split("|")))/5000*elements_count_penalty
                    print(f"Effectiveness & Efficiency Score: {ee * 100:.3f}%")
                    if best_ee < ee:
                        best_ee = ee
                        win_option = o
                        win_expr = reg.last_solve_expression
                        opt_precision = precision
                        opt_recall = recall
                        opt_f1 = f1
            if best_ee_sofar < best_ee * 0.985:
                ct_opt = ct_now
                best_ee_sofar = best_ee
                ct_now = ct_now + jump
                win_option_sofar = win_option
                expr_opt = win_expr
                opt_precision_sofar = opt_precision
                opt_recall_sofar = opt_recall
                opt_f1_sofar = opt_f1
            elif jump == 1:
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
                print(self.replaceSegName(expr_opt))
                print("")
                print("#################################")
                print("")
                break
            else:
                jump = int(jump/2)
                ct_now = ct_now - jump





##### SAMPLE EXECUTION #########


import copy
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

def measure(start_time):
    end_time = time.time()

    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    milliseconds = int(elapsed_time * 1000 % 1000)
    return f"{int(minutes)}:{str(int(seconds)).zfill(2)}:{str(int(milliseconds)).zfill(3)}"
           
start_time = time.time()


###### Load the breast cancer dataset ###### 
data = load_breast_cancer()
X, y = data.data, data.target
X_list = [list(a) for a in X]
y_list = list(y)
heads = [f.replace(" ", "_") for f in data.feature_names]
X_list.insert(0, heads)
y_list.insert(0, "Result")
data_list = [X_list[i] + [y_list[i]] for i in range(len(X_list))]

reg = Deterministic_Regressor()
inp = reg.train(data_list=data_list, error_tolerance=0.03, max_dnf_len=3, by_four=1, min_match=30)

print("train() finished", measure(start_time))
start_time = time.time()# # print(inp)


answer = [int(inp[i][-1]) for i in range(1, len(inp), 1)]
inp = [row[:-1] for row in inp]

reg.optimize_params(inp, answer)
#reg.optimize_params(inp, answer, used_options=["union", "common"])


