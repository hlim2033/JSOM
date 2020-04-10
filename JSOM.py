import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances

class JSOM(object):
    def __init__(self, data1, data2, matching1, matching2, m, n, n_iterations, ratio=None, alpha=None, sigma=None):
        # data1 should be a np array of size [# of samples , # of features ]
        # data2 should be a np array of size [# of samples , # of features ]
        # matching1 is a np array of size [# of samples for data1, ]. Each element, Xi, 
                            #is the index of its(i-th data from data1) closest data in data2 
        # matching2 is a np array of size [# of samples for data2 ]. Each element, Xi, 
                            #is the index of its(i-th data from data2) closest data in data1
        # m = # of SOM nodes in x-axis 
        # n = # of SOM nodes in y-axis 
        # n_iteration = # of times it fits 
        
        #Assign required variables first
        self.Data1 = data1
        self.Data2 = data2
        self.Matching1to2 = matching1
        self.Matching2to1 = matching2
        self.m = m
        self.n = n
        self.num_runs = n_iterations
        if alpha is None:
            self.alpha = 0.9
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = np.sqrt(2)*(np.maximum(m,n)-1)/3
        else:
            self.sigma = float(sigma)
        
        if ratio is None:
            self.ratio = 4
        else:
            self.ratio = int(ratio)
        
        self.NN = m*n
        
    def train(self):    
        self.nodes_coord = []
        ct = 0
        for i in range(self.m):
            for j in range(self.n):
                ct = ct + 1
                self.nodes_coord.append([i,j])
        self.nodes_coord = np.array(self.nodes_coord)
        
        random_index_1 = np.random.randint(0, len(self.Data1)-1, size=self.NN)
        random_index_2 = np.random.randint(0, len(self.Data2)-1, size=self.NN)

        ##random initiazliation 
        permData1 = np.random.permutation(np.random.permutation(self.Data1).T).T
        permData2 = np.random.permutation(np.random.permutation(self.Data2).T).T

        ct = 0
        self.nodes_vectors_array_1 = []
        self.nodes_vectors_array_2 = []
        for i in range(self.NN):
            ct = ct + 1
            self.nodes_vectors_array_1.append(permData1[random_index_1[i],:])
            self.nodes_vectors_array_2.append(permData2[random_index_2[i],:])
        self.nodes_vectors_array_1 = np.array(self.nodes_vectors_array_1, dtype=float)
        self.nodes_vectors_array_2 = np.array(self.nodes_vectors_array_2, dtype=float)
        self.initial_nodes_vectors_array_1 = self.nodes_vectors_array_1[:]
        self.initial_nodes_vectors_array_2 = self.nodes_vectors_array_2[:]
        input_order1 = np.random.permutation(len(self.Data1))
        input_order2 = np.random.permutation(len(self.Data2))
        input_order3 = np.random.permutation(len(self.Data1))
        input_order4 = np.random.permutation(len(self.Data2))
        
        ct_d1 = 0 
        ct_d2 = 0
        ct_d3 = 0
        ct_d4 = 0
            
        for run in range(self.num_runs):
            if run % 3000 == 0:
                print('Training...', run, 'out of', self.num_runs)
            e = self.sigma*(1-run/self.num_runs)
            if run < np.ceil(0.2 * self.num_runs):
                alpha = self.alpha
            else:
                alpha = self.alpha*(run-(np.ceil(0.2 * self.num_runs)-1))*(-1/(self.num_runs-(np.ceil(0.2 * self.num_runs)-1)))+1
            #coupled 
            if run % self.ratio == 0:
                ct_d1 = ct_d1 + 1
                input_from_1 = np.array(self.Data1[input_order1[ct_d1-1],:], dtype = float)
                input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
                distance_to_nodes = euclidean_distances(input_from_1, self.nodes_vectors_array_1)
                nn = np.argmin(distance_to_nodes)
                mapping_index = self.Matching1to2[input_order1[ct_d1-1]]
                im_input_for2 = self.Data2[mapping_index,:]
                for node in range(self.NN):
                    a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                    b = np.reshape(self.nodes_coord[nn], (1,len(self.nodes_coord[nn])))
                    if euclidean_distances(a,b)[0] < e:
                        self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + alpha * (input_from_1 - self.nodes_vectors_array_1[node,:])
                        self.nodes_vectors_array_2[node,:] = self.nodes_vectors_array_2[node,:] + alpha * (im_input_for2 - self.nodes_vectors_array_2[node,:])

                ct_d2 = ct_d2 + 1
                input_from_2 = np.array(self.Data2[input_order2[ct_d2-1],:], dtype = float)
                input_from_2 = np.reshape(input_from_2, (1,len(input_from_2)))
                distance_to_nodes = euclidean_distances(input_from_2, self.nodes_vectors_array_2)
                nn = np.argmin(distance_to_nodes)
                mapping_index = self.Matching2to1[input_order2[ct_d2-1]]
                im_input_for1 = self.Data1[mapping_index,:]
                for node in range(self.NN):
                    a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                    b = np.reshape(self.nodes_coord[nn], (1,len(self.nodes_coord[nn])))
                    if euclidean_distances(a,b)[0] < e:
                        self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + alpha * (im_input_for1 - self.nodes_vectors_array_1[node,:])
                        self.nodes_vectors_array_2[node,:] = self.nodes_vectors_array_2[node,:] + alpha * (input_from_2 - self.nodes_vectors_array_2[node,:])

            else:
                ct_d3 = ct_d3 + 1
                input_from_1 = np.array(self.Data1[input_order3[ct_d3-1],:], dtype = float)
                input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
                distance_to_nodes = euclidean_distances(input_from_1, self.nodes_vectors_array_1)
                nn = np.argmin(distance_to_nodes)
                for node in range(self.NN):
                    a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                    b = np.reshape(self.nodes_coord[nn], (1,len(self.nodes_coord[nn])))
                    if euclidean_distances(a,b)[0] < e:
                        self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + alpha * (input_from_1 - self.nodes_vectors_array_1[node,:])

                ct_d4 = ct_d4 + 1
                input_from_2 = np.array(self.Data2[input_order4[ct_d4-1],:], dtype = float)
                input_from_2 = np.reshape(input_from_2, (1,len(input_from_2)))
                distance_to_nodes = euclidean_distances(input_from_2, self.nodes_vectors_array_2)
                nn = np.argmin(distance_to_nodes)
                for node in range(self.NN):
                    a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                    b = np.reshape(self.nodes_coord[nn], (1,len(self.nodes_coord[nn])))
                    if euclidean_distances(a,b)[0] < e:
                        self.nodes_vectors_array_2[node,:] = self.nodes_vectors_array_2[node,:] + alpha * (input_from_2 - self.nodes_vectors_array_2[node,:])
            if ct_d1 > len(self.Data1)-1:
                ct_d1 = 0
            if ct_d2 > len(self.Data2)-1:
                ct_d2 = 0
            if ct_d3 > len(self.Data1)-1:
                ct_d3 = 0
            if ct_d4 > len(self.Data2)-1:
                ct_d4 = 0
        print('training_done')
        
    def initial_vectors(self):
        return self.initial_nodes_vectors_array_1, self.initial_nodes_vectors_array_2

    def nodes_weights(self):
        return self.nodes_vectors_array_1, self.nodes_vectors_array_2

    def mapping(self):
        NODES_1 = []
        for i in range(len(self.Data1)):
            input_from_1 = np.array(self.Data1[i,:], dtype = float)
            input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
            distance_to_nodes = euclidean_distances(input_from_1, self.nodes_vectors_array_1)
            nn = np.argmin(distance_to_nodes)
            NODES_1.append(nn)

        NODES_2 = []
        for i in range(len(self.Data2)):
            input_from_2 = np.array(self.Data2[i,:], dtype = float)
            input_from_2 = np.reshape(input_from_2, (1,len(input_from_2)))
            distance_to_nodes = euclidean_distances(input_from_2, self.nodes_vectors_array_2)
            nn = np.argmin(distance_to_nodes)
            NODES_2.append(nn)
        NODES_1 = np.array(NODES_1)
        NODES_2 = np.array(NODES_2)        
        SUMMARY = np.zeros((1,4))
        
        self.Data1_nodes = []
        self.Data2_nodes = []
        for i in range(self.NN):
            summary = np.zeros((1,4))
            index1 = np.where(NODES_1 == i)[0];
            index2 = np.where(NODES_2 == i)[0];
            summary[0,0]=len(index1)
            summary[0,1]=len(index2)
            if len(index1)+len(index2) == 0:
                summary[0,2] = 0
                summary[0,3] = 0
            else:
                summary[0,2] = len(index1)/(len(index1) + len(index2))
                summary[0,3] = len(index2)/(len(index1) + len(index2))
            #print(summary)
            SUMMARY = np.vstack((SUMMARY,summary))
            self.Data1_nodes.append(index1)
            self.Data2_nodes.append(index2)
        SUMMARY = SUMMARY[1:,:]
        ENTROPY = []
        for i in range(self.NN):
            if SUMMARY[i,2] == 0 or SUMMARY[i,3] == 0:
                entropy = 0
            else:
                entropy=-SUMMARY[i,2]*np.log2(SUMMARY[i,2])-SUMMARY[i,3]*np.log2(SUMMARY[i,3])
            ENTROPY.append(entropy)

        self.High_E_Index = np.where(np.array(ENTROPY) > 0.2)[0]        
        self.NODE_LOC_1to2 = []
        IDX = []
        for i in range(len(self.Data1)):
            for j in range(len(self.Data1_nodes)):
                node_cell = self.Data1_nodes[j]
                if i in node_cell:
                    node_loc = self.Data2_nodes[j]
                    distance = 1.1
                    while len(node_loc) == 0:
                        neighbors = []
                        for node in range(self.NN):
                            a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                            b = np.reshape(self.nodes_coord[j], (1,len(self.nodes_coord[j])))
                            if euclidean_distances(a,b)[0] < distance and len(self.Data2_nodes[node]) > 0:
                                for element in range(len(self.Data2_nodes[node])):
                                    neighbors.append(self.Data2_nodes[node][element])                  
                        node_loc = np.array(neighbors)
                        distance = distance + 1
                    self.NODE_LOC_1to2.append(node_loc)                   
        
        self.NODE_LOC_2to1 = []
        IDX = []
        for i in range(len(self.Data2)):
            for j in range(len(self.Data2_nodes)):
                node_loc = self.Data2_nodes[j]
                if i in node_loc:
                    node_cell = self.Data1_nodes[j]
                    distance = 1.1
                    while len(node_cell) == 0:
                        neighbors = []
                        for node in range(self.NN):
                            a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                            b = np.reshape(self.nodes_coord[j], (1,len(self.nodes_coord[j])))
                            if euclidean_distances(a,b)[0] < distance and len(self.Data1_nodes[node]) > 0:
                                for element in range(len(self.Data1_nodes[node])):
                                    neighbors.append(self.Data1_nodes[node][element])                  
                        node_cell = np.array(neighbors)
                        distance = distance + 1
                    self.NODE_LOC_2to1.append(node_cell)
        print('Mapping Done')
        return self.NODE_LOC_1to2, self.NODE_LOC_2to1, self.High_E_Index 
   
    def data_per_nodes(self):
        return self.Data1_nodes, self.Data2_nodes


class SOM(object):
    def __init__(self, data, m, n, n_iterations, alpha=None, sigma=None):
        # data should be a np array of size [# of samples , # of features ]
        # m = # of SOM nodes in x-axis 
        # n = # of SOM nodes in y-axis 
        # n_iteration = # of times it fits 
        
        self.Data1 = data
        self.m = m
        self.n = n
        self.num_runs = n_iterations
        if alpha is None:
            self.alpha = 0.9
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = np.sqrt(2)*(np.maximum(m,n)-1)/3
        else:
            self.sigma = float(sigma)
        self.ratio = 3
        self.NN = m*n
        
    def som_train(self):    
        self.nodes_coord = []
        ct = 0
        for i in range(self.m):
            for j in range(self.n):
                ct = ct + 1
                self.nodes_coord.append([i,j])
        self.nodes_coord = np.array(self.nodes_coord)
        
        random_index_1 = np.random.randint(0, len(self.Data1)-1, size=self.NN)
        
        ct = 0
        self.nodes_vectors_array_1 = []
        for i in range(self.NN):
            ct = ct + 1
            self.nodes_vectors_array_1.append(self.Data1[random_index_1[i],:])
        self.nodes_vectors_array_1 = np.array(self.nodes_vectors_array_1, dtype=float)
        
        input_order1 = np.random.permutation(len(self.Data1))
        
        ct_d1 = 0 
            
        for run in range(self.num_runs):
            if run % 3000 == 0:
                print(run)
            e = self.sigma*(1-run/self.num_runs)
            if run < np.round(0.3*self.num_runs):
                alpha = self.alpha
            else:
                alpha = self.alpha*(run-np.round(0.3*self.num_runs))*(-1/self.num_runs)+1
            #coupled 

            ct_d1 = ct_d1 + 1
            input_from_1 = np.array(self.Data1[input_order1[ct_d1-1],:], dtype = float)
            input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
            distance_to_nodes = euclidean_distances(input_from_1, self.nodes_vectors_array_1)
            nn = np.argmin(distance_to_nodes)
            for node in range(self.NN):
                a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                b = np.reshape(self.nodes_coord[nn], (1,len(self.nodes_coord[nn])))
                if euclidean_distances(a,b)[0] < e:
                    self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + alpha * (input_from_1 - self.nodes_vectors_array_1[node,:])
            
            if ct_d1 > len(self.Data1)-1:
                ct_d1 = 0
        print('training_done')
        
    def som_nodes_weights(self):
        return self.nodes_vectors_array_1
    
    def mapping(self):
        NODES_1 = []
        for i in range(len(self.Data1)):
            input_from_1 = np.array(self.Data1[i,:], dtype = float)
            input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
            distance_to_nodes = euclidean_distances(input_from_1, self.nodes_vectors_array_1)
            nn = np.argmin(distance_to_nodes)
            NODES_1.append(nn)
            
        Data1_nodes = []
        for i in range(self.NN):
            index1 = np.where(np.array(NODES_1) == i)[0];
            Data1_nodes.append(index1)
            
        return NODES_1, Data1_nodes

