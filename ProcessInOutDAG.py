# Process In-Out DAG
# Version: 2.0.2
# Last Update: 2024/02/24
# Author: Tomio Kobayashi
#
# pip install simpy

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random
from datetime import date
import textwrap
import re
from scipy.sparse import csr_matrix
import simpy

class ProcessInOutDAG:

    def __init__(self):
        self.vertex_names = []
        self.csr_matrix = []
        self.csr_matrix_T = []
        self.size_matrix = 0
        
        self.dic_vertex_names = {}
        self.dic_vertex_id = {}
        
        self.G = 0
        self.G_T = 0
        
        self.str_vertex_names = []
        self.str_csr_matrix = []
        self.str_csr_matrix_T = []
        self.str_size_matrix = 0
        
        self.str_dic_vertex_names = {}
        self.str_dic_vertex_id = {}
        
        self.str_G = 0
        self.str_G_T = 0
        
        self.dic_new2old = {}
        self.dic_old2new = {}

        self.set_complete = set()
        
        self.dic_conds = {}
        self.dic_opts = {}
        
        self.avg_duration = {}
        self.max_pos = 0
        
        
        self.simpy_env = simpy.Environment()
        self.flow_counter = 0
        self.task_finished = {}
        self.task_triggered = {}
        self.start_times = {}
        self.finish_times = {}
        self.sim_nodesets = set()
        
    # SIMULATOR 
    
    def myeval(self, __ccc___, __inp___):
        for __jjjj___ in __ccc___:
#             print("__jjjj___", __jjjj___)
            exec(__jjjj___[0] + " = " + str(__jjjj___[1]))
        return eval(__inp___)

    def task(self, target_vertex, flow_seq, workers, silent=False):
        
        if flow_seq not in self.task_triggered:
            self.task_triggered[flow_seq] = set()
        if flow_seq not in self.task_finished:
            self.task_finished[flow_seq] = set()
            
        if target_vertex in self.task_triggered[flow_seq]:
            return
        
        
#         preds_set = set(self.G.predecessors(target_vertex))
        preds_set = set([p for p in self.G.predecessors(target_vertex) if p in self.sim_nodesets])
        
        if self.dic_vertex_names[target_vertex] in self.dic_conds:
            s = [(p, True if p in self.task_finished[flow_seq] else False) for p in preds_set]
            ss = self.dic_conds[self.dic_vertex_names[target_vertex]].replace("&", " and ").replace("|", " or ")
            res = self.myeval([(self.dic_vertex_names[p], True if p in self.task_finished[flow_seq] else False) for p in preds_set], self.dic_conds[self.dic_vertex_names[target_vertex]].replace("&", " and ").replace("|", " or "))
#             print("res", res)
            if not res:
                return
        else:
            for p in preds_set:
                if p != target_vertex and p not in self.task_finished[flow_seq]:
                    return

        self.task_triggered[flow_seq].add(target_vertex)
            
        succs_set = set(self.G.successors(target_vertex))
        if len(succs_set) == 0:
            if not silent:
#                 print(self.dic_vertex_names[target_vertex])
                print("FLOW", flow_seq, f"COMPLETED AT {self.simpy_env.now:.2f}")
#             print("self.task_finished", self.task_finished)
            return
        weight = max([self.G[target_vertex][s]["weight"] for s in list(succs_set)])
#         time_takes = max(np.random.normal(weight, min(weight, 3)), 0.5)
        time_takes = np.log(np.random.lognormal(weight, min(weight, 3))+1)
            
        start_time = self.simpy_env.now
        if target_vertex not in self.start_times:
            self.start_times[target_vertex] = []
        self.start_times[target_vertex].append(start_time)

        yield self.simpy_env.timeout(time_takes)
        
        finish_time = self.simpy_env.now
        if target_vertex not in self.finish_times:
            self.finish_times[target_vertex] = []
        self.finish_times[target_vertex].append(finish_time)
        if not silent:
            print(self.dic_vertex_names[target_vertex], "for", flow_seq, f"fiinished at {finish_time:.2f}")
        
        self.task_finished[flow_seq].add(target_vertex)
        
        
        # Start process and run
        for s in list(succs_set):
            workers = simpy.Resource(self.simpy_env, capacity=9999999)
            self.simpy_env.process(self.task(s, flow_seq, workers, silent=silent))
            
    def start_flow(self, target_vertices, silent=False, sim_repeats=1, fromSink=True):
        
        dic_target_vertices = {}
        for target_vertex in target_vertices:
            if isinstance(target_vertex, str):
                if target_vertex not in self.dic_vertex_id:
                    f = [v for k, v in self.dic_vertex_id.items() if target_vertex == re.sub("(#C.*)", "", k, count=1)]
#                     f = [re.sub("(#C.*)", "", k, count=1) for k, v in self.dic_vertex_id.items() if target_vertex == re.sub("(#C.*)", "", k, count=1)]
                    if len(f) > 0:
#                         dic_target_vertices[target_vertex] = f[0]
                        dic_target_vertices[re.sub("(#C.*)", "", target_vertex, count=1)] = f[0]
                    else:
                        print(target_vertex + " is not an element")
                        return
                else:
                    dic_target_vertices[target_vertex] = self.dic_vertex_id[target_vertex]
                
        
        # Set up and start the simulation
        print("")
        print('Flow Simulation Started')
#         random.seed(42)  # For reproducible results
        
        self.flow_counter = 0
        self.task_finished = {}
        self.task_triggered = {}
        self.start_times = {}
        self.finish_times = {}
        
        self.sim_nodesets = set()
        
        if fromSink:
#             tmp_target_vetcices = []
            tmp_target_vetcices = set()
            for target_vertex in target_vertices:
                self.sim_nodesets |= set(self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=dic_target_vertices[target_vertex], orientation="reverse"))]).nodes)
#                 tmp_target_vetcices += [node for node, in_degree in self.G.in_degree() if in_degree == 0]
                tmp_target_vetcices != set([node for node, in_degree in self.G.in_degree() if in_degree == 0])
            dic_target_vertices = {self.dic_vertex_names[t]:t for t in tmp_target_vetcices}
            target_vertices = [self.dic_vertex_names[t] for t in tmp_target_vetcices]
        else:
            for target_vertex in target_vertices:
                self.sim_nodesets |= set(self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=dic_target_vertices[target_vertex]))]).nodes)
        
#         print("target_vertices", target_vertices)
#         print("dic_target_vertices", dic_target_vertices)
            
        for i in range(sim_repeats):
#             print("i", i)
            self.simpy_env = simpy.Environment()
            for target_vertex in target_vertices:
                # Start process and run
                workers = simpy.Resource(self.simpy_env, capacity=9999999)
                self.simpy_env.process(self.task(dic_target_vertices[target_vertex], self.flow_counter, workers, silent=silent))
            self.flow_counter += 1
            self.simpy_env.run()
        
        print("----------------")
        for k, v in self.start_times.items():
#             print(self.dic_vertex_names[k], f"starts at {np.mean(v):.2f}, and finishes at {np.mean(self.finish_times[k]):.2f} in average")
            print(re.sub("(#C.*)", "", self.dic_vertex_names[k], count=1), f"starts at {np.mean(v):.2f} and finishes at {np.mean(self.finish_times[k]):.2f} in average")
        print("")
            
    def csr_matrix_to_edge_list(self, csr_matrix):
        rows, cols = csr_matrix.nonzero()
        weights = csr_matrix.data
        edge_list = list(zip(rows, cols, weights))
        return edge_list
        
        
    def adjacency_matrix_to_edge_list(self, adc_matrix):
        edge_list = []
        num_nodes = len(adc_matrix)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if adc_matrix[i][j] >= 1:
                    edge_list.append((i, j, adc_matrix[i][j]))  # Add edges only for non-zero entries

        return edge_list
    
    def parameter_to_brightness(self, parameter_value):
        cmap = plt.cm.get_cmap('viridis')  # Choose a colormap
        brightness = cmap(parameter_value)[0]  # Extract first value (red channel)
        return brightness
    
    def find_pos(self, subgraph, use_expected=True, sigma=3, use_weight_one=False, use_lognormal=True):
        
        cum_duration = {}
        avg_duration = {}
        avg_duration_r = {}
        topological_order = None
        the_graph = None
        
        topological_order = list(nx.topological_sort(subgraph))
        the_graph = subgraph

        weights = {}
        for i, j in the_graph.edges():
            if j not in weights:
                weights[j] = {}
            weights[j][self.dic_vertex_names[i]] = the_graph[i][j]['weight']

        for t in topological_order:
            if t not in weights:
                avg_duration[t] = 0
                cum_duration[t] = 0
            else:
                tot = 0
                weight_params = {k: avg_duration[self.dic_vertex_id[k]] + v for k, v in weights[t].items()} if not use_weight_one else {k: avg_duration[self.dic_vertex_id[k]] + 1 for k, v in weights[t].items()} 
#                 normal_sigma = min(max([v for k, v in weight_params.items()]), sigma)
                normal_sigma = min(max([v for k, v in weights[t].items()]), sigma)
                if use_expected:
                    if self.dic_vertex_names[t] in self.dic_conds:
#                         tot = logical_weight.calc_avg_result_weight(self.dic_conds[self.dic_vertex_names[t]], weight_params, opt_steps=self.dic_opts, use_lognormal=False, normal_sigma=normal_sigma)
                        tot = logical_weight.calc_avg_result_weight(self.dic_conds[self.dic_vertex_names[t]], weight_params, opt_steps=self.dic_opts, use_lognormal=use_lognormal, sigma=normal_sigma)
                    else:
#                         tot = logical_weight.calc_avg_result_weight(" & ".join([k for k, v in weights[t].items()]), weight_params, opt_steps=self.dic_opts, use_lognormal=False, normal_sigma=normal_sigma)
                        tot = logical_weight.calc_avg_result_weight(" & ".join([k for k, v in weights[t].items()]), weight_params, opt_steps=self.dic_opts, use_lognormal=use_lognormal, sigma=normal_sigma)
                else:
                    tot = max([v for k, v in weight_params.items()])
                avg_duration[t] = tot
                weight_params_cum = {k: cum_duration[self.dic_vertex_id[k]] + v for k, v in weights[t].items()}
                cum_duration[t] = max([v for k, v in weight_params_cum.items()])
        max_duration = max([v for k, v in avg_duration.items()])
        avg_duration_r = {k: max_duration - v for k, v in avg_duration.items()}
        self.avg_duration = avg_duration
        weight_list = [(self.dic_vertex_id[kk], k, round(avg_duration[k] - avg_duration[self.dic_vertex_id[kk]] - vv, 1)) for k, v in weights.items() for kk, vv in v.items() if int(avg_duration[k] - avg_duration[self.dic_vertex_id[kk]]) - vv > 0]

        max_pos = max_duration
        
        self.max_pos = max_duration
#         pos = {k: (int(np.round(avg_duration[k], 0)), 0) for k in the_graph.nodes}
        div = 1 if use_weight_one or len(avg_duration) < 15 or not use_expected else 2
        pos = {k: (int(np.round(avg_duration[k]/div, 0)), 0) for k in the_graph.nodes}

        last_pos = max([v[0] for k, v in pos.items()])
        colpos = {l:0 for l in range(last_pos+1)}
        for k, v in pos.items():
            colpos[v[0]] += 1
        dicPos = {i: 0 for i in range(len(colpos))}
        for k, v in sorted(pos.items(), reverse=True):
            pos[k] = (v[0], dicPos[v[0]])
            dicPos[v[0]] += 1


#         re-align the vertical position
        maxheight = max([v for k, v in colpos.items() if v != 0])
        newpos = {}
        for k, v in pos.items():
            gap = (maxheight) / colpos[v[0]]
            newheight = (gap/2) + v[1]*gap
            newpos[k] = (v[0], newheight)

#         change orders to minimize line crossings
        for posx in sorted(list(set([v[0] for k, v in newpos.items()]))):
#             for iii in range(1):
            for iii in range(int(colpos[posx]/2+1)):
                for node in [k for k, v in newpos.items() if v[0] == posx]:
                    incoming_edges = the_graph.in_edges(node)

                    predecessors = [edge[0] for edge in incoming_edges]
                    pred_in_pos = list(set([p for p in predecessors if p in newpos]))
                    if len(pred_in_pos) > 0:
                        pred_heights = [newpos[p][1] for p in pred_in_pos]
                        avg_pred_heights = average = sum(pred_heights) / len(pred_heights)
                        closest_node = 0
                        closest_height = 9999
                        closest_dist = 9999
                        [my_pos, my_height] = newpos[node]
                        my_dist = np.abs(my_height - avg_pred_heights)
                        for k, v in newpos.items():
                            if k == node:
                                continue
                            if v[0] == my_pos:
                                dist = np.abs(v[1] - avg_pred_heights)
                                if dist < closest_dist:
                                    closest_node = k
                                    closest_dist = dist
                                    closest_height = v[1]
                        if closest_dist < my_dist:
                            newpos[closest_node] = (my_pos, my_height)
                            newpos[node] = (my_pos, closest_height)

        pos = newpos
        
        return newpos, weight_list

    def draw_selected_vertices_reverse_proc2(self, G, selected_vertices1, selected_vertices2, selected_vertices3, title, node_labels, 
                                            pos, wait_edges=None, reverse=False, figsize=(12, 8), showWeight=False, forStretch=False, excludeComp=False, showExpectationBased=False):
        
        pattern = "(#C.*)"
        
        comp_vertices = None
        subgraph_comp = None
        noncomp_vertices = None
        subgraph_noncomp = None
        if excludeComp:
            if reverse:
                excluded_tup = [(f[0], f[1]) for s in self.set_complete for f in list(nx.edge_dfs(G, source=self.dic_vertex_id[s]))]
            else:
                excluded_tup = [(f[0], f[1]) for s in self.set_complete for f in list(nx.edge_dfs(G, source=self.dic_vertex_id[s], orientation="reverse"))]
            excluded = set([f[0] for f in excluded_tup] + [f[1] for f in excluded_tup])
            comp_vertices = [s for s in selected_vertices1 if s in excluded]
            noncomp_vertices = [s for s in selected_vertices1 if s not in excluded]
            subgraph_comp = G.subgraph(comp_vertices)
            subgraph_noncomp = G.subgraph(noncomp_vertices)
            
        # Create a subgraph with only the selected vertices
        subgraph1 = G.subgraph(selected_vertices1)
        subgraph2 = G.subgraph(selected_vertices2)
        subgraph3 = G.subgraph(selected_vertices3)
        
        if reverse:
            subgraph1 = subgraph1.reverse()
            subgraph2 = subgraph2.reverse()
            subgraph3 = subgraph3.reverse()
            if excludeComp:
                subgraph_comp = subgraph_comp.reverse()
                subgraph_noncomp = subgraph_noncomp.reverse()

        # Set figure size to be larger
        plt.figure(figsize=figsize)

#         # Set figure size to be larger
#         node_labels = {node: "\n".join(["\n".join(textwrap.wrap(s, width=5)) for s in label.replace("_", "_\n").replace(" ", "\n").split("\n")]) for node, label in node_labels.items()}
    
        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0
        
        longest_path_length = None
        
        if showWeight:

            longest_path_length = None
            if not excludeComp:
                longest_path_length = nx.dag_longest_path_length(subgraph1)
            else:
                longest_path_length = nx.dag_longest_path_length(subgraph_noncomp)
                
            node_criticality = None
            if longest_path_length == 0:
                node_criticality = []
            else:
                if has_proc:
                    node_criticality = [((nx.dag_longest_path_length(subgraph1.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(subgraph1, target_node))])) + 
                                    nx.dag_longest_path_length(subgraph1.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(subgraph1, target_node, orientation='reverse'))]))) / 
                                    longest_path_length, target_node) for target_node in subgraph1.nodes() if self.dic_vertex_names[target_node][0:5] == "proc_"]
                else:
                    node_criticality = [((nx.dag_longest_path_length(subgraph1.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(subgraph1, target_node))])) + 
                                    nx.dag_longest_path_length(subgraph1.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(subgraph1, target_node, orientation='reverse'))]))) / 
                                    longest_path_length, target_node) for target_node in subgraph1.nodes()]
                
            node_parameter = {n[1]: round(n[0]*0.7+0.3, 3) for n in node_criticality} # make it between 0.2 and 1.0 to prevent too dark color
            if len(node_parameter) > 0:
                color_map = [plt.cm.viridis(node_parameter[node]) for node in subgraph2]

        if forStretch and len(selected_vertices1) > 10:
            max_height = max([v[1] for k, v in pos.items()])
            for k, v in pos.items():
                if v[0] % 3 == 1:
#                     pos[k] = (v[0], v[1] + max_height * 0.05)
                    pos[k] = (v[0], v[1] + max_height * 0.1)
                if v[0] % 3 == 2:
#                     pos[k] = (v[0], v[1] - max_height * 0.05)
                    pos[k] = (v[0], v[1] - max_height * 0.1)
                    
        # Draw the graph
        if has_proc and forStretch:
            defFontSize=8
            defNodeSize=700
            defFontColor='grey'
        else:
            defFontSize=10
            defNodeSize=1000
            defFontColor='black'

        cum_duration = {}
        avg_duration = {}
        avg_duration_r = {}
        topological_order = None
        the_graph = None
        if showWeight:
            # Find the topological order
            if not excludeComp:
                topological_order = list(nx.topological_sort(subgraph1))
                the_graph = subgraph1
            else:
                topological_order = list(nx.topological_sort(subgraph_noncomp))
                the_graph = subgraph_noncomp
                    
        node_labels1 = {k: v for k, v in node_labels.items() if k in subgraph1 and k not in subgraph2 and k not in subgraph3}
        node_colors = ['skyblue' for n in selected_vertices1]
        for i, s in enumerate(selected_vertices1):
#             if s in node_labels1 and node_labels1[s][-3:-1] == "#C":
#                 node_colors[i] = node_labels1[s][-2:]
#                 node_labels1[s] = node_labels1[s][:-3]
            if s in node_labels1:
                match = re.search(pattern, node_labels1[s])
                if match:
                    matched_str = str(match.group()[2:])
#                     print("node_labels1[s]", node_labels1[s])
#                     print("matched_str", matched_str)
#                     print("match.group()", match.group())
#                     print("match.group()[2:]", match.group()[2:])
                    node_colors[i] = matched_str if len(matched_str) > 1 else "C" + match.group()[2:]
                    node_labels1[s] = re.sub(pattern, "", node_labels1[s], count=1)
    
        for i in node_labels:
#             if node_labels[i][-3:-1] == "#C":
#                 node_labels[i] = node_labels[i][:-3]
            match = re.search(pattern, node_labels[i])
            if match:
                node_labels[i] = re.sub(pattern, "", node_labels[i], count=1)
                
        
        # Set figure size to be larger
        node_labels = {node: "\n".join(["\n".join(textwrap.wrap(s, width=5)) for s in label.replace("_", "_\n").replace(" ", "\n").split("\n")]) for node, label in node_labels.items()}
        node_labels1 = {node: "\n".join(["\n".join(textwrap.wrap(s, width=5)) for s in label.replace("_", "_\n").replace(" ", "\n").split("\n")]) for node, label in node_labels1.items()}

        
        if showWeight and forStretch:
            node_labels1 = {k: v + "\n(" + str(round(self.avg_duration[k], 1)) + ")" for k, v in node_labels1.items()}
            node_labels2 = {k: v + "\n(" + str(round(self.avg_duration[k], 1)) + ")" for k, v in node_labels.items() if k in subgraph2}
            node_labels3 = {k: v + "\n(" + str(round(self.avg_duration[k], 1)) + ")" for k, v in node_labels.items() if k in subgraph3}
        else:
            node_labels1 = {k: v for k, v in node_labels1.items()}
            node_labels2 = {k: v for k, v in node_labels.items() if k in subgraph2}
            node_labels3 = {k: v for k, v in node_labels.items() if k in subgraph3}

        if forStretch and showExpectationBased:
            pos = {k: (self.avg_duration[k], v[1])for k, v in pos.items()}
            
        nx.draw(subgraph1, pos, linewidths=0, with_labels=True, labels=node_labels1, node_size=defNodeSize, nodelist=selected_vertices1, node_color=node_colors, font_size=defFontSize, font_color=defFontColor, arrowsize=10, edgecolors='black')

        if has_proc and showWeight and len(node_parameter) > 0:
            nx.draw(subgraph2, pos, linewidths=0, with_labels=True, labels=node_labels2, node_size=defNodeSize, node_color=color_map, font_size=10, font_color='black', arrowsize=10, edgecolors='black')
        else:
            nx.draw(subgraph2, pos, linewidths=0, with_labels=True, labels=node_labels2, node_size=defNodeSize, node_color='orange', font_size=10, font_color='black', arrowsize=10, edgecolors='black')

        nx.draw(subgraph3, pos, linewidths=0, with_labels=True, labels=node_labels3, node_size=defNodeSize, node_color='pink', font_size=10, font_color='black', arrowsize=10, edgecolors='black')
        
        if excludeComp:
            nx.draw(subgraph_comp, pos, linewidths=0, with_labels=True, labels=node_labels1, node_size=defNodeSize, node_color='#DDDDDD', font_size=defFontSize, font_color=defFontColor, arrowsize=10, edgecolors='black')
    
        if showWeight:
            if wait_edges is None:
                edge_labels = {(i, j): subgraph1[i][j]['weight'] for i, j in subgraph1.edges()}
            else:
                dicWait = {(w[0], w[1]): w[2] for w in wait_edges}
                edge_labels = {(i, j): subgraph1[i][j]['weight'] if (i, j) not in dicWait else str(subgraph1[i][j]['weight']) + " (& " + str(dicWait[(i, j)]) + ")"  for i, j in subgraph1.edges()}

            nx.draw_networkx_edge_labels(subgraph1, pos, edge_labels=edge_labels)
        
        if wait_edges is not None:
        
            wait_edge_list = [(w[0], w[1]) for w in wait_edges if w[2] < 5]
            nx.draw_networkx_edges(subgraph1, pos, edgelist=wait_edge_list, edge_color='aqua', width=1.00)
            wait_edge_list = [(w[0], w[1]) for w in wait_edges if w[2] >= 5]
            nx.draw_networkx_edges(subgraph1, pos, edgelist=wait_edge_list, edge_color='aqua', width=1.00)
            
        # Draw critical path edges with a different color
        if showWeight:
            longest_path = None
            if not excludeComp:
                longest_path_length = nx.dag_longest_path_length(subgraph1)
                longest_path = nx.dag_longest_path(subgraph1)  # Use NetworkX's built-in function
                critical_edges = [(longest_path[i], longest_path[i + 1]) for i in range(len(longest_path) - 1)]
                nx.draw_networkx_edges(subgraph1, pos, edgelist=critical_edges, edge_color='brown', width=1.25)
            else:
                longest_path_length = nx.dag_longest_path_length(subgraph_noncomp)
                longest_path = nx.dag_longest_path(subgraph_noncomp)  # Use NetworkX's built-in function
                critical_edges = [(longest_path[i], longest_path[i + 1]) for i in range(len(longest_path) - 1)]
                nx.draw_networkx_edges(subgraph_noncomp, pos, edgelist=critical_edges, edge_color='brown', width=1.25)

        if forStretch:
            # Draw a vertical line at x=0.5 using matplotlib
            max_horizontal_pos = int(max([v[0] for k, v in pos.items()]))

            for i in range(0, max_horizontal_pos, 5):
                width = 0.25
                linestyle = "dashed" if (max_horizontal_pos - i) % 10 == 0 else "dotted"
                plt.axvline(x=i, color="orange", linestyle=linestyle, linewidth=width)

        if showWeight:
            
            

            self.showStats(subgraph1)
            # Print the topological order
            print("TOPOLOGICAL ORDER:")
            print(" > ".join([re.sub(pattern, "", self.dic_vertex_names[t], count=1) for t in topological_order if (has_proc and self.dic_vertex_names[t][0:5] == "proc_") or not has_proc]))

            print("")

            # Print the longest path and its length
            print("LONGEST PATH (" + str(longest_path_length) + "):")
            print(" > ".join([re.sub(pattern, "", self.dic_vertex_names[t], count=1) for t in longest_path if (has_proc and self.dic_vertex_names[t][0:5] == "proc_") or not has_proc]))
            print("")
#             print("CRITICALITY:")
#             for n in sorted(node_criticality, reverse=True):
#                 print(self.dic_vertex_names[n[1]], round(n[0], 3))
#             print("")
            # Print the longest path and its length

    
            if reverse==False and forStretch:
                print("REALISTIC EXPECTED COMPLETION TIME")
                print(", ".join([re.sub(pattern, "", self.dic_vertex_names[t], count=1) + ": " + str(round(self.avg_duration[t], 1)) for t in topological_order]))
                print("")
    
            self.suggest_coupling(subgraph1)
            self.suggest_opportunities(subgraph1)

            
        plt.title("Expecation Based - " + title if showExpectationBased else title)
        plt.show()
    
    

    def edge_list_to_csr_matrix(self, edges):
        rows = [edge[0] for edge in edges]
        cols = [edge[1] for edge in edges]
        weights = [edge[2] for edge in edges]
        num_nodes = max(max(rows), max(cols))+1
        matrix = csr_matrix((weights, (rows, cols)), shape=(num_nodes, num_nodes))
        return matrix

    
    def edge_list_to_adjacency_matrix(self, edges):

        num_nodes = max(max(edge) for edge in edges) + 1  # Determine the number of nodes
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for edge in edges:
            if len(edge) >= 3:
                adjacency_matrix[edge[0]][edge[1]] = edge[2]
            else:
                adjacency_matrix[edge[0]][edge[1]] = 1

        return adjacency_matrix

    def data_import(self, file_path=None, is_edge_list=False, is_oup_list=False, intext=""):
#         Define rows as TO and columns as FROM
        adjacency_matrix = None
        if is_oup_list:
            edges = self.read_oup_list_from_file(file_path, intext=intext)
            adjacency_matrix = self.edge_list_to_adjacency_matrix(edges)
        elif is_edge_list:
            edges = self.read_edge_list_from_file(file_path, intext=intext)
            adjacency_matrix = self.edge_list_to_adjacency_matrix(edges)
        else:
            data = np.genfromtxt(file_path, delimiter='\t', dtype=str, encoding=None)

            # Extract the names from the first record
            self.vertex_names = list(data[0])

            # Extract the adjacency matrix
            adjacency_matrix = data[1:]
        
        # Convert the adjacency matrix to a NumPy array of integers
        adjacency_matrix = np.array(adjacency_matrix, dtype=int)
                    
        if not nx.is_directed_acyclic_graph(nx.DiGraph(adjacency_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(adjacency_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        
        self.csr_matrix = csr_matrix(adjacency_matrix)
        self.csr_matrix_T = self.csr_matrix.transpose()

        # Matrix of row=FROM, col=TO
        self.size_matrix = len(adjacency_matrix)

        self.G = nx.DiGraph(self.csr_matrix)
        self.G_T = nx.DiGraph(self.csr_matrix_T)
        
        for i in range(len(self.vertex_names)):
            self.dic_vertex_names[i] = self.vertex_names[i]
            self.dic_vertex_id[self.vertex_names[i]] = i

        # Find the largest connected component
        connected_components = list(nx.weakly_connected_components(self.G))
        
        largest_connected_component = None
        print("Sizes of Connected Graphs")
        print("---")
        sorted_graphs = sorted([[len(c), c] for c in connected_components], reverse=True)
        for i in range(len(sorted_graphs)):
            print(sorted_graphs[i][0])
            if i == 0:
                largest_connected_component = sorted_graphs[i][1]
            
        largest_G = self.G.subgraph(largest_connected_component)
        
        print("")
        # Show stats of procs
        print("Centrality Stats of the Largest Connected Component")
        print("---")
        self.showStats(largest_G)
            
    def write_edge_list_to_file(self, filename, intext=""):
        
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)

        with open(filename, "w") as file:
            # Write headers
            file.write("\t".join(self.vertex_names) + "\n")
            for edge in edges:
                file.write(f"{edge[0]}\t{edge[1]}\t{edge[2]}\n")
    
    def read_oup_list_from_file(self, filename=None, intext=""):
#         Format is tab-delimited taxt file:
#             0 - Output Field Name
#             1 - Weight
#             2-n - Input Field Names
        dictf = {}
        dicfields = {}
        lines = []
        if intext == "":
            with open(filename, "r") as file:
                for line in file:
                    lines.append(line)
        else:
            lines = intext.split("\n")
        
        lines = [line for line in lines if len(line) > 0]
        
        ini = True
        for line in lines:
            ff = line.strip().split("\t")
            for i in [0] + [j for j in range(2, len(ff), 1)]:
                if ff[i] not in dicfields:
                    dicfields[ff[i]] = len(dicfields)

            oup_ind = dicfields[ff[0]]
            if dicfields[ff[0]] not in dictf:
                inp = [dicfields[ff[i]] for i in range(2, len(ff), 1)]
                dictf[oup_ind] = (int(ff[1]), inp)
            else:
                dictf[oup_ind][1].extend([dicfields[ff[i]] for i in range(2, len(ff), 1)]) 

        headers = [k[1] for k in sorted([(v, k) for k, v in dicfields.items()])]
        edges = [(vv, k, v[0]) for k, v in dictf.items() for vv in v[1]]

        self.vertex_names = headers
        return edges
    
    def read_complete_list_from_file(self, filename=None, intext=""):
#         with open(filename, "r") as file:
#             for line in file:
                
        lines = []
        if intext == "":
            with open(filename, "r") as file:
                for line in file:
                    lines.append(line)
        else:
            lines = intext.split("\n")
        
        lines = [line for line in lines if len(line) > 0]
        
        for line in lines:
            s = line.strip()
            self.set_complete.add(s)
            if s[0:5] != "proc_":
                self.set_complete.add("proc_" + s)
                    
    def read_cond_list_from_file(self, filename=None, intext=""):
#         with open(filename, "r") as file:
#             for line in file:
        lines = []
        if intext == "":
            with open(filename, "r") as file:
                for line in file:
                    lines.append(line)
        else:
            lines = intext.split("\n")
        
        lines = [line for line in lines if len(line) > 0]
        
        for line in lines:

            conds = line.strip().split("\t")
            self.dic_conds[conds[0]] = conds[1]

    def read_opt_list_from_file(self, filename=None, intext=""):
#         with open(filename, "r") as file:
#             for line in file:
        lines = []
        if intext == "":
            with open(filename, "r") as file:
                for line in file:
                    lines.append(line)
        else:
            lines = intext.split("\n")
        
        lines = [line for line in lines if len(line) > 0]
        
        for line in lines:
            conds = line.strip().split("\t")
            self.dic_opts[conds[0]] = float(conds[1])

    def read_edge_list_from_file(self, filename=None, intext=""):
#         edges = []
#         with open(filename, "r") as file:
#             ini = True
#             for line in file:

        lines = []
        if intext == "":
            with open(filename, "r") as file:
                for line in file:
                    lines.append(line)
        else:
            lines = intext.split("\n")
        
        lines = [line for line in lines if len(line) > 0]
        edges = []
        ini = True
        for line in lines:
            if ini:
                self.vertex_names = line.strip().split("\t")
                ini = False
            else:
                source, target, weight = map(int, line.strip().split())
                edges.append((source, target, weight))
        return edges

    def write_adjacency_matrix_to_file(self, filename):
        with open(filename, "w") as file:
            file.write("\t".join(self.vertex_names) + "\n") 
            for row in self.csr_matrix.toarray():
                file.write("\t".join(map(str, row)) + "\n")  # Join elements with tabs and add newline


    def genProcesses(self):
        
        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k[0:5] == "proc_"]) > 0
        if has_proc:
            print("Already contains processes.")
            return

                
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)
        new_edges = {}
        dicNewID = {}
        setVertex = set(self.vertex_names)


        dicEdgeWeight = {}
        for k in edges:
            if k[0] not in dicEdgeWeight or (k[0] in dicEdgeWeight and dicEdgeWeight[k[0]] < k[2]):
                dicEdgeWeight[k[0]] = k[2]
        for i in range(len(edges)):
            new_vertex_name = "proc_" + self.dic_vertex_names[edges[i][1]]
            new_id = 0
            if new_vertex_name not in setVertex:
                new_id = len(self.vertex_names)
                np.append(self.vertex_names, new_vertex_name)
                self.vertex_names.append(new_vertex_name)
                self.dic_vertex_names[new_id] = new_vertex_name
                self.dic_vertex_id[new_vertex_name] = new_id
                dicNewID[edges[i][1]] = new_id
                setVertex.add(new_vertex_name)
            else:
                new_id = dicNewID[edges[i][1]]
            w = 1
            if edges[i][1] in dicEdgeWeight: 
                w = dicEdgeWeight[edges[i][1]]
            if (edges[i][0], new_id) not in new_edges:
                new_edges[(edges[i][0], new_id)] = 1
            if (new_id, edges[i][1]) not in new_edges or ((new_id, edges[i][1]) in new_edges and w * 2 - 1 > new_edges[(new_id, edges[i][1])]):
                new_edges[(new_id, edges[i][1])] = w * 2 - 1
            
        new_edges = [(k[0], k[1], v) for k, v in new_edges.items()]
    
        self.csr_matrix = self.edge_list_to_csr_matrix(new_edges)
        self.csr_matrix_T = self.csr_matrix.transpose()
        self.size_matrix = self.csr_matrix.shape[0]
        
        
        self.G = nx.DiGraph(self.csr_matrix)
        self.G_T = nx.DiGraph(self.csr_matrix_T)
        
        # Show stats of procs
                
        print("")

        # Find the largest connected component
        connected_components = list(nx.weakly_connected_components(self.G))
        largest_connected_component = None
        sorted_graphs = sorted([[len(c), c] for c in connected_components], reverse=True)
        for i in range(len(sorted_graphs)):
            if i == 0:
                largest_connected_component = sorted_graphs[i][1]
        largest_G = self.G.subgraph(largest_connected_component)
        
        print("")
        # Show stats of procs
        print("Centrality Stats of the Largest Connected Component")
        print("---")
        self.showStats(largest_G)
        
    def coupleProcesses(self, proc1, proc2):
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)
        if proc1[0:5] != "proc_" or proc2[0:5] != "proc_":
            return
        new_edges = []
        old = self.dic_vertex_id[proc1]
        new = self.dic_vertex_id[proc2]
        
        lowers = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, old))])
        uppers = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, orientation='reverse'))])
        if new in lowers and new in uppers:
            print("Processes that lie on the same path cannot be coupled.", proc1, proc2)
            return
        
        newWeight = max([edges[i][2] for i in range(len(edges)) if edges[i][0] == old]) + max([edges[i][2] for i in range(len(edges)) if edges[i][0] == new])

        for i in range(len(edges)):
            id1 = edges[i][0]
            id2 = edges[i][1]
            weight = edges[i][2]
            if edges[i][0] == old:
                id1 = new
                weight = newWeight
            if edges[i][0] == new:
                id1 = new
                weight = newWeight
            if edges[i][1] == old:
                id2 = new
            
            new_edges.append((id1, id2, weight))
    
        tmp_matrix = self.edge_list_to_csr_matrix(new_edges)
                   
        if not nx.is_directed_acyclic_graph(nx.DiGraph(tmp_matrix)):       
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(tmp_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        else:
            self.csr_matrix = tmp_matrix
            self.csr_matrix_T = self.csr_matrix.transpose()
            self.size_matrix = self.csr_matrix.shape[0]
        
            self.G = nx.DiGraph(self.csr_matrix)
            self.G_T = nx.DiGraph(self.csr_matrix_T)

    def create_random_string_from_date(self):

        today = date.today()
        year_str = str(today.year)
        month_str = str(today.month).zfill(2)  # Pad month with leading zero if needed
        day_str = str(today.day).zfill(2)  # Pad day with leading zero if needed

        date_parts = list(year_str + month_str + day_str)
        random.shuffle(date_parts)  # Shuffle the digits randomly

        random_string = "".join(date_parts[:5])  # Take the first 5 digits
        return random_string


    def linkElements(self, element1, element2, weight, procName=""):
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)
        isProcessIncluded = (self.dic_vertex_names[edges[1][0]][0:5] == "proc_" or self.dic_vertex_names[edges[1][1]][0:5] == "proc_")
        if element1[0:5] == "proc_" or element1[0:5] == "proc_":
            return
        new_edges = copy.deepcopy(edges)
        if element1 not in self.dic_vertex_id or element2 not in self.dic_vertex_id:
            return
        id1 = self.dic_vertex_id[element1]
        id2 = self.dic_vertex_id[element2]
        
        newID = 99999
        if isProcessIncluded:
            if procName == "":
                procName = "proc_" + element2 + "_" + self.create_random_string_from_date()
            newID = max([max(e[0], e[1]) for e in edges]) + 1
            new_edges.append((id1, newID, 1))
            new_edges.append((newID, id2, weight))
            self.vertex_names.append(procName)
            self.dic_vertex_names[newID] = procName
            self.dic_vertex_id[procName] = newID
        else:
            new_edges.append((id1, id2, weight))
    
        tmp_matrix = self.edge_list_to_csr_matrix(new_edges)
        
        if not nx.is_directed_acyclic_graph(nx.DiGraph(tmp_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(tmp_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        else:
            self.csr_matrix_T = self.csr_matrix.transpose()
            self.size_matrix = self.csr_matrix.shape[0]
        
            self.G = nx.DiGraph(self.csr_matrix)
            self.G_T = nx.DiGraph(self.csr_matrix_T)
            if isProcessIncluded:
                self.vertex_names.append(procName)
                self.dic_vertex_names[newID] = procName
                self.dic_vertex_id[procName] = newID
            

        
    def drawOriginsStretch(self, target_vertex, title="", figsize=None, showWeight=False, excludeComp=False, showExpectationBased=False, use_lognormal=True):

#         if isinstance(target_vertex, str):
#             if target_vertex not in self.dic_vertex_id:
#                 print(target_vertex + " is not an element")
#                 return
#             target_vertex = self.dic_vertex_id[target_vertex]
            
        if isinstance(target_vertex, str):
            if target_vertex not in self.dic_vertex_id:
                f = [v for k, v in self.dic_vertex_id.items() if target_vertex == re.sub("(#C.*)", "", k, count=1)]
                if len(f) > 0:
                    target_vertex = f[0]
                else:
                    print(target_vertex + " is not an element")
                    return
            else:
                target_vertex = self.dic_vertex_id[target_vertex]
        
        # Draw the path TO the target
        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        
        subgraph = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=target_vertex, orientation="reverse"))])
        succs = [self.dic_vertex_names[s] for s in subgraph]

        position, wait_edges = self.find_pos(subgraph, use_expected=showExpectationBased, use_lognormal=use_lognormal)
       
        selected_vertices1 = set([n for n in subgraph.nodes])
        selected_vertices2 = set([n for n in subgraph.nodes if self.dic_vertex_names[n][0:5] == "proc_"])

        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Origins with Weighted Pipelining"
#             title += " (" + str(max([v[0] for k, v in position.items()])) + " steps)"
            title += " (" + str(int(self.max_pos)) + " steps)"

        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]
        
        if figsize is None:
            figsize = (12, 8)
        
        self.draw_selected_vertices_reverse_proc2(self.G, selected_vertices1,selected_vertices2, selected_vertices3, 
                        title=title, node_labels=node_labels, pos=position, figsize=figsize, showWeight=showWeight, forStretch=True, wait_edges=wait_edges, excludeComp=excludeComp, 
                                                  showExpectationBased=showExpectationBased)
        
    def drawOrigins(self, target_vertex, title="", figsize=None, showWeight=False, excludeComp=False, use_lognormal=True):

#         if isinstance(target_vertex, str):
#             if target_vertex not in self.dic_vertex_id:
#                 print(target_vertex + " is not an element")
#                 return
#             target_vertex = self.dic_vertex_id[target_vertex]
    
        if isinstance(target_vertex, str):
            if target_vertex not in self.dic_vertex_id:
                f = [v for k, v in self.dic_vertex_id.items() if target_vertex == re.sub("(#C.*)", "", k, count=1)]
                if len(f) > 0:
                    target_vertex = f[0]
                else:
                    print(target_vertex + " is not an element")
                    return
            else:
                target_vertex = self.dic_vertex_id[target_vertex]
                
        # Draw the path TO the target
        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        
        subgraph = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=target_vertex, orientation="reverse"))])
        succs = [self.dic_vertex_names[s] for s in subgraph]

        position, wait_edges = self.find_pos(subgraph, use_expected=False, use_weight_one=True, use_lognormal=use_lognormal)
       
        selected_vertices1 = set([n for n in subgraph.nodes])
        selected_vertices2 = set([n for n in subgraph.nodes if self.dic_vertex_names[n][0:5] == "proc_"])

        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Origins"
#             title += " (" + str(max([v[0] for k, v in position.items()])) + " steps)"
        title += " (" + str(int(self.max_pos)) + " steps)"
 
        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]
        
        if figsize is None:
            figsize = (12, 8)

        self.draw_selected_vertices_reverse_proc2(self.G, selected_vertices1,selected_vertices2, selected_vertices3, 
                                title=title, node_labels=node_labels, pos=position, figsize=figsize, showWeight=showWeight, excludeComp=excludeComp)
    

    def drawOffspringsStretch(self, target_vertex, title="", figsize=None, showWeight=False, excludeComp=False, showExpectationBased=False, use_lognormal=True):

#         if isinstance(target_vertex, str):
#             if target_vertex not in self.dic_vertex_id:
#                 print(target_vertex + " is not an element")
#                 return
#             target_vertex = self.dic_vertex_id[target_vertex]

        if isinstance(target_vertex, str):
            if target_vertex not in self.dic_vertex_id:
                f = [v for k, v in self.dic_vertex_id.items() if target_vertex == re.sub("(#C.*)", "", k, count=1)]
                if len(f) > 0:
                    target_vertex = f[0]
                else:
                    print(target_vertex + " is not an element")
                    return
            else:
                target_vertex = self.dic_vertex_id[target_vertex]

#         Draw the path FROM the target
#         position = {}
#         colpos = {}
#         posfill = set()
#         selected_vertices1 = set()
#         selected_vertices2 = set()
            
#         # Draw the path TO the target
        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        

        
        subgraph = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=target_vertex))])
        succs = [self.dic_vertex_names[s] for s in subgraph]

        position, wait_edges = self.find_pos(subgraph, use_expected=showExpectationBased, use_lognormal=use_lognormal)
       
        selected_vertices1 = set([n for n in subgraph.nodes])
        selected_vertices2 = set([n for n in subgraph.nodes if self.dic_vertex_names[n][0:5] == "proc_"])

        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Offsprings with Weighted Pipelining"
#         title += " (" + str(max([v[0] for k, v in position.items()])) + " steps)"
        title += " (" + str(int(self.max_pos)) + " steps)"

        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]
        
        if figsize is None:
            figsize = (12, 8)
        
        self.draw_selected_vertices_reverse_proc2(self.G_T, selected_vertices1,selected_vertices2, selected_vertices3, 
                        title=title, node_labels=node_labels, pos=position, reverse=True, figsize=figsize, showWeight=showWeight, forStretch=True, wait_edges=wait_edges, 
                                                 excludeComp=excludeComp, showExpectationBased=showExpectationBased)
        
    def drawOffsprings(self, target_vertex, title="", figsize=None, showWeight=False, excludeComp=False, use_lognormal=True):

#         if isinstance(target_vertex, str):
#             if target_vertex not in self.dic_vertex_id:
#                 print(target_vertex + " is not an element")
#                 return
#             target_vertex = self.dic_vertex_id[target_vertex]

        if isinstance(target_vertex, str):
            if target_vertex not in self.dic_vertex_id:
                f = [v for k, v in self.dic_vertex_id.items() if target_vertex == re.sub("(#C.*)", "", k, count=1)]
                if len(f) > 0:
                    target_vertex = f[0]
                else:
                    print(target_vertex + " is not an element")
                    return
            else:
                target_vertex = self.dic_vertex_id[target_vertex]
                
#         Draw the path FROM the target
#         position = {}
#         colpos = {}
#         posfill = set()
#         selected_vertices1 = set()
#         selected_vertices2 = set()
            
#         # Draw the path TO the target
        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        

        subgraph = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=target_vertex))])
        succs = [self.dic_vertex_names[s] for s in subgraph]
        position, wait_edges = self.find_pos(subgraph, use_expected=False, use_weight_one=True, use_lognormal=use_lognormal)
       
        selected_vertices1 = set([n for n in subgraph.nodes])
        selected_vertices2 = set([n for n in subgraph.nodes if self.dic_vertex_names[n][0:5] == "proc_"])
        selected_vertices3 = [target_vertex]

        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Offsprings"
#         title += " (" + str(max([v[0] for k, v in position.items()])) + " steps)"
        title += " (" + str(int(self.max_pos)) + " steps)"

        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]
        
        if figsize is None:
            figsize = (12, 8)

        self.draw_selected_vertices_reverse_proc2(self.G_T, selected_vertices1,selected_vertices2, selected_vertices3, 
                        title=title, node_labels=node_labels, pos=position, reverse=True, figsize=figsize, showWeight=showWeight, excludeComp=excludeComp)
        

    
        
    def showSourceNodes(self):
        
        re_pattern = "(#C.*)"
        
        sources = [node for node in self.G.nodes() if len(list(self.G.predecessors(node))) == 0]

#         print("Source Nodes:", [self.dic_vertex_names[s] for s in sources])
        print("Source Nodes:", [re.sub(re_pattern, "", self.dic_vertex_names[s], count=1) for s in sources])
        
        
    def showSinkNodes(self):
        
        re_pattern = "(#C.*)"
        
        sinks = [node for node in self.G.nodes() if len(list(self.G.successors(node))) == 0]

#         print("Sink Nodes:", [self.dic_vertex_names[s] for s in sinks])
        print("Sink Nodes:", [re.sub(re_pattern, "", self.dic_vertex_names[s], count=1) for s in sinks])
        
    def showStats(self, g):
        
        re_pattern = "(#C.*)"
        
        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0
        if has_proc and list(nx.weakly_connected_components(g)) == 1:
            is_bipartite = nx.bipartite.sets(g)
            if is_bipartite:
                self.showBipartiteStats(g)
                return
            
        in_degree_centrality = nx.in_degree_centrality(g)
        out_degree_centrality = nx.out_degree_centrality(g)
#         closeness_centrality = nx.closeness_centrality(g)
#         betweenness_centrality = nx.betweenness_centrality(g)
        
        cnt_max = 5
        print("")
        print("In Degree Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in in_degree_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
            print(re.sub(re_pattern, "", self.dic_vertex_names[z[1]], count=1), round(z[0], 3))
            cnt += 1
        print("")
        
        print("Out Degree Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in out_degree_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
            print(re.sub(re_pattern, "", self.dic_vertex_names[z[1]], count=1), round(z[0], 3))
            cnt += 1
        print("")
        
#         print("Closeness Centrality:")
#         cnt = 0
#         for z in sorted([(v, k) for k, v in closeness_centrality.items()], reverse=True):
#             if cnt == cnt_max:
#                 break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
#             cnt += 1
#         print("")
#         print("Betweenness Centrality:")
#         cnt = 0
#         for z in sorted([(v, k) for k, v in betweenness_centrality.items()], reverse=True):
#             if cnt == cnt_max:
#                 break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
#             cnt += 1
#         print("")
        
    def showBipartiteStats(self, g):
        is_bipartite, node_sets = nx.bipartite.sets(g)
        projection = nx.bipartite.projected_graph(g, node_sets)
        degree_centrality = nx.degree_centrality(projection)
#         closeness_centrality = nx.closeness_centrality(projection)
#         betweenness_centrality = nx.betweenness_centrality(projection)
        cnt_max = 5
        print("Degree Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in degree_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 3))
            cnt += 1
        print("")
#         print("Closeness Centrality:")
#         cnt = 0
#         for z in sorted([(v, k) for k, v in closeness_centrality.items()], reverse=True):
#             if cnt == cnt_max:
#                 break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
#             cnt += 1
#         print("")
#         print("Betweenness Centrality:")
#         cnt = 0
#         for z in sorted([(v, k) for k, v in betweenness_centrality.items()], reverse=True):
#             if cnt == cnt_max:
#                 break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
#             cnt += 1
#         print("")
        
    def drawFromLargestComponent(self, figsize=(30, 30), showWeight=False, excludeComp=False):
        
        connected_components = list(nx.weakly_connected_components(self.G))
        largest_connected_component = None
        sorted_graphs = sorted([[len(c), c] for c in connected_components], reverse=True)
        for i in range(len(sorted_graphs)):
            if i == 0:
                largest_connected_component = sorted_graphs[i][1]
        largest_G = self.G.subgraph(largest_connected_component)
        
        # Find the topological order
        topological_order = list(nx.topological_sort(largest_G))
        self.drawOffsprings(topological_order[0], figsize=figsize, showWeight=showWeight, excludeComp=excludeComp)
        self.drawOrigins(topological_order[-1], figsize=figsize, showWeight=showWeight, excludeComp=excludeComp)
        
    
    def suggest_coupling(self, g):
        re_pattern = "(#C.*)"
        longest_path_length = nx.dag_longest_path_length(g)
        
        if longest_path_length == 0:
            return
        
        longest_path = nx.dag_longest_path(g)
        node_criticality = None
        node_criticality = [((nx.dag_longest_path_length(g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, target_node))])) + 
                        nx.dag_longest_path_length(g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, target_node, orientation='reverse'))]))) / 
                        longest_path_length, target_node) for target_node in g.nodes() if self.dic_vertex_names[target_node][0:5] == "proc_"]

        
        print("SUGGESTED COUPLINGS")
        cnt = 0
        for i in range(len(node_criticality)):
            if cnt > 20:
                break
            if node_criticality[i][1] in longest_path:
                continue
            for j in range(i+1, len(node_criticality), 1):
                if node_criticality[j][1] in longest_path:
                    continue
                if node_criticality[i][0] + node_criticality[j][0] > 1.3:
                    continue;
                lowers = g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, node_criticality[i][1]))])
                uppers = g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, node_criticality[i][1], orientation='reverse'))])
                if node_criticality[j][1] not in lowers and node_criticality[j][1] not in uppers:

                    lowers_j = g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, node_criticality[j][1]))])
                    diff_longest = np.abs(nx.dag_longest_path_length(lowers) - nx.dag_longest_path_length(lowers_j))
                    if diff_longest < 6:
#                         print(self.dic_vertex_names[node_criticality[i][1]], self.dic_vertex_names[node_criticality[j][1]])
                        print(re.sub(re_pattern, "", self.dic_vertex_names[node_criticality[i][1]], count=1), re.sub(re_pattern, "", self.dic_vertex_names[node_criticality[j][1]], count=1))
                        
                        cnt += 1
                        if cnt > 20:
                            break
                            

        
        
        print("")
        
    def suggest_opportunities(self, g):
        
        re_pattern = "(#C.*)"
        longest_path_length = nx.dag_longest_path_length(g)

        opportunity_list = []
        for n in g.nodes():
            if self.dic_vertex_names[n][0:5] != "proc_":
                continue

            h = g.copy()
            outgoing_edges = list(h.out_edges(n))
            for e in outgoing_edges:
                h[e[0]][e[1]]["weight"] = 1      
            improvement_without = longest_path_length - nx.dag_longest_path_length(h)
            if improvement_without > 1:
                opportunity_list.append((improvement_without, n))

        print("SUGGESTED THROUGHPUT IMPROVEMENT OPPORTUNITIES")
        for n in sorted(opportunity_list, reverse=True):
#             print(self.dic_vertex_names[n[1]], n[0])
            print(re.sub(re_pattern, "", self.dic_vertex_names[n[1]], count=1), n[0])
            
        print("")

    def getMatrix(self):
        return self.csr_matrix.toarray()
    
    def getVertexNames(self):
        return self.vertex_names
    
    def getDicVertexNames(self):
        return self.dic_vertex_names
  

    def keepOnlyProcesses(self):
        
        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0
        if not has_proc:
            print("There are no processe nodes in the graph.")
            return
        connected_components = list(nx.weakly_connected_components(self.G))
        largest_connected_component = None
        if len(connected_components) > 0:
            print("Keeping only the largest connected graph.")
            sorted_graphs = sorted([[len(c), c] for c in connected_components], reverse=True)
            for i in range(len(sorted_graphs)):
                if i == 0:
                    largest_connected_component = sorted_graphs[i][1]
            largest_G = self.G.subgraph(largest_connected_component)
        
            is_bipartite = nx.bipartite.sets(largest_G)
            if not is_bipartite:
                print("There is not a bipartite graph.")
                return
        
            self.G =  largest_G
        else:
            is_bipartite = nx.bipartite.sets(self.G)
            if not is_bipartite:
                print("There is not a bipartite graph.")
                return
        
        
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)
        new_edges = []
        for i in range(len(edges)):
            id1 = edges[i][0]
            id2 = edges[i][1]
            if self.dic_vertex_names[id1][0:5] != "proc_":
                continue
            
            w = self.G[id1][id2]["weight"]
                
            for s in list(self.G.successors(id2)):
                w2 = self.G[id2][s]["weight"]
                weight = int((w + w2)/2)
                if weight == 0: weight = 1
                new_edges.append((id1, s, weight))
        tmp_matrix = self.edge_list_to_csr_matrix(new_edges)

        if not nx.is_directed_acyclic_graph(nx.DiGraph(tmp_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(tmp_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
        else:
            self.csr_matrix = tmp_matrix
            self.csr_matrix_T = self.csr_matrix.transpose()
            self.size_matrix = self.csr_matrix.shape[0]
        
            self.G = nx.DiGraph(self.csr_matrix)
            self.G_T = nx.DiGraph(self.csr_matrix_T)
            
            for i in range(len(copy.deepcopy(self.vertex_names))):
                if i not in self.G.nodes:
                    self.vertex_names.pop(i)
                    
                    