import torch
import itertools

from models.SGM1.affinity_layer import InnerProductWithWeightsAffinity
from models.SGM1.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from lpmp_py import GraphMatchingModule
from lpmp_py import MultiGraphMatchingModule
from src.feature_align import feature_align
from src.lap_solvers.sinkhorn import Sinkhorn
from torch.nn.utils.rnn import pad_sequence
from stableMatching.sm_solvers.stable_marriage import stable_marriage
from IPython.core.debugger import Tracer

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


def lexico_iter(lex):
    return itertools.combinations(lex, 2)

def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.SGM1.FEATURE_CHANNEL)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.SGM1.FEATURE_CHANNEL
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)
        self.rescale = cfg.PROBLEM.RESCALE
        self.sinkhorn = Sinkhorn(max_iter=cfg.SGM1.SK_ITER_NUM, epsilon=cfg.SGM1.SK_EPSILON, tau=cfg.SGM1.SK_TAU)

    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        num_graphs = len(images)
        
        

        if cfg.PROBLEM.TYPE == '2GM' and 'gt_perm_mat' in data_dict:
            gt_perm_mats = [data_dict['gt_perm_mat']]
        else:
            raise ValueError('Ground truth information is required during training.')

        global_list = []
        orig_graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = torch.cat((U, F), dim=1)
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)
            
            
            
       
        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]
        
        sinkhorn_similarities = []
        
        
        
        
        for (g_1, g_2), global_weights,(ns_src, ns_tgt) in zip(lexico_iter(orig_graph_list), global_weights_list,lexico_iter(n_points)):
            similarity=self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            ns_srcs = ns_src
            ns_tgts = ns_tgt
            
            
            
            for i in range(len(similarity)):
                if(ns_srcs[i]==ns_tgts[i]):
                    s = self.sinkhorn(similarity[i], None, None,True)
                else:
                    s = self.sinkhorn(similarity[i], ns_srcs[i], ns_tgts[i],True)
                    Tracer()()
                sinkhorn_similarities.append(s)
                    
        # Determine maximum length
        max_len_0 = max([x.size(dim=0) for x in sinkhorn_similarities])
        max_len_1= max([x.size(dim=1) for x in sinkhorn_similarities])
        # pad all tensors to have same length
        sinkhorn_similarities = [torch.nn.functional.pad(x, pad=(0, max_len_0 - x.size(dim=0),0,max_len_1-x.size(dim=1)), mode='constant', value=0) for x in sinkhorn_similarities]
        
        
       
        ss = torch.stack(sinkhorn_similarities)
       
    
        perm_mat_np =stable_marriage(ss, n_points[0], n_points[1])
        data_dict.update({
            'ds_mat': ss,
            'cs_mat': ss,
            'perm_mat':perm_mat_np,
        })
        
        return data_dict
            
            
        
