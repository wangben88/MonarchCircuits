from __future__ import annotations

import torch
import math
import numpy as np
import networkx as nx
import pickle
from typing import Type, Optional

from pyjuice.nodes.distributions import *

from pyjuice.nodes import multiply, summate, inputs, CircuitNodes
from pyjuice.utils.util import max_cdf_power_of_2
    
def BayesianTreeToHiddenRegionGraphGeneral(tree: nx.Graph, 
                                    root,
                                    num_latents: int,  
                                    InputDist: Type[Distribution], 
                                    dist_params: dict,
                                    layer_fn: Type[CircuitNodes],
                                    num_root_ns: int = 1,
                                    homogeneous_inputs: bool = True,
                                    homog_by_channel: bool = False,
                                    block_size: Optional[int] = None,
                                    permute_block_size: Optional[int] = None,
                                    input_params: torch.Tensor = None,
                                    num_latents_schedule=None,
                                    ) -> CircuitNodes:
    """
    Given a Tree Bayesian Network tree T1 (i.e. at most one parents), 
    
        1. Bayesian Network T1 becomes `tree` rooted at `root`
        2. Construct Bayesian bayesian network T2:
            - T2 = copy of T1
            - In T2, convert each variable x_i to a categoriacal latent variable z_i with `num_latents` categories
            - In T2, Adds back x_i nodes, with one edge x_i -> z_i 

        3. Returns a RegionGraph of a probabilistics circuit that is equivalent to T2

    For example, if T1 = "x1 -> x2" , then T2  becomes  x1    x2 
                                                        ^     ^
                                                        |     |
                                                        z1 -> z2   
    """
    # Root the tree at `root`
    clt = nx.bfs_tree(tree, root)
    def children(n: int):
        return [c for c in clt.successors(n)]
    
    # Assert at most one parent
    for n in clt.nodes:
        assert len(list(clt.predecessors(n))) <= 1

    # Compile the region graph for the circuit equivalent to T2
    node_seq = list(nx.dfs_postorder_nodes(tree, root))
    var2rnode = dict()
    leaf_nodes = dict()
    ns_sum = None

    img_dim = int(math.sqrt(len(node_seq) // 3))
    if block_size is None:
        block_size = min(1024, max_cdf_power_of_2(num_latents))
    num_node_blocks = num_latents // block_size

    for v in node_seq:
        leaf_idx = v//(img_dim ** 2) if homog_by_channel else 0
        chs = children(v)

        if len(chs) == 0:
            # Input Region
            if leaf_idx not in leaf_nodes:
                if input_params is not None:
                    r = inputs(v, num_node_blocks = num_node_blocks, block_size=block_size, dist = InputDist(**dist_params), params = input_params)
                else:
                    r = inputs(v, num_node_blocks = num_node_blocks, block_size=block_size, dist = InputDist(**dist_params))
                leaf_nodes[leaf_idx] = r
            else:
                r = leaf_nodes[leaf_idx].duplicate(v, tie_params=homogeneous_inputs)
            var2rnode[v] = r
        else:
            # Inner Region
            
            # children(z_v)
            ch_regions = [var2rnode[c] for c in chs]

            # Add x_v to children(z_v)
            if leaf_idx not in leaf_nodes:
                if input_params is not None:
                    leaf_r = inputs(v, num_node_blocks = num_node_blocks, block_size=block_size, dist = InputDist(**dist_params), params = input_params)
                else:
                    leaf_r = inputs(v, num_node_blocks = num_node_blocks, block_size=block_size, dist = InputDist(**dist_params))
                leaf_nodes[leaf_idx] = leaf_r
            else:
                leaf_r = leaf_nodes[leaf_idx].duplicate(v, tie_params=homogeneous_inputs)
            ch_regions.append(leaf_r)

            if v == root:
                assert num_root_ns == 1, "Only one root node allowed"
                rp = multiply(*ch_regions)
                r = summate(rp, num_node_blocks = num_root_ns, block_size = 1)
            else:
                r, ns_sum = layer_fn(*ch_regions, num_node_blocks = num_node_blocks, block_size = block_size,
                                            permute_block_size = permute_block_size, existing_nodes = ns_sum, homogeneous = False,
                                            num_nodes = num_latents)

            var2rnode[v] = r
    root_r = var2rnode[root]
    return root_r


def mutual_information(x1: torch.Tensor, x2: torch.Tensor, num_bins: int, sigma: float):
    assert x1.device == x2.device

    device = x1.device
    B, K1 = x1.size()
    K2 = x2.size(1)

    x1 = (x1 - torch.min(x1)) / (torch.max(x1) - torch.min(x1) + 1e-8)
    x2 = (x2 - torch.min(x2)) / (torch.max(x2) - torch.min(x2) + 1e-8)

    bins = torch.linspace(0, 1, num_bins, device = device)

    x1p = torch.exp(-0.5 * (x1.unsqueeze(2) - bins.view(1, 1, -1)).pow(2) / sigma**2) # (B, K1, n_bin)
    x2p = torch.exp(-0.5 * (x2.unsqueeze(2) - bins.view(1, 1, -1)).pow(2) / sigma**2) # (B, K2, n_bin)

    x12p = torch.einsum("bia,baj->ij", x1p.reshape(B, K1 * num_bins, 1), x2p.reshape(B, 1, K2 * num_bins)).reshape(K1, num_bins, K2, num_bins) / B

    x1p_norm = (x1p / x1p.sum(dim = 2, keepdim = True)).mean(dim = 0)
    x2p_norm = (x2p / x2p.sum(dim = 2, keepdim = True)).mean(dim = 0)
    x12p_norm = x12p / x12p.sum(dim = (1, 3), keepdim = True) # (K1, n_bin, K2, n_bin)

    m1 = -(x1p_norm * torch.log(x1p_norm + 1e-4)).sum(dim = 1)
    m2 = -(x2p_norm * torch.log(x2p_norm + 1e-4)).sum(dim = 1)
    m12 = -(x12p_norm * torch.log(x12p_norm + 1e-4)).sum(dim = (1, 3))

    mi = m1.unsqueeze(1) + m2.unsqueeze(0) - m12
    return mi


def mutual_information_chunked(x1: torch.Tensor, x2: torch.Tensor, num_bins: int, sigma: float, chunk_size: int):
    K = x1.size(1)
    mi = torch.zeros([K, K])
    for x_s in range(0, K, chunk_size):
        x_e = min(x_s + chunk_size, K)
        for y_s in range(0, K, chunk_size):
            y_e = min(y_s + chunk_size, K)

            mi[x_s:x_e,y_s:y_e] = mutual_information(x1[:,x_s:x_e], x2[:,y_s:y_e], num_bins, sigma)

    return mi


def chow_liu_tree(mi: np.ndarray):
    K = mi.shape[0]
    G = nx.Graph()
    for v in range(K):
        G.add_node(v)
        for u in range(v):
            G.add_edge(u, v, weight = -mi[u, v])

    T = nx.minimum_spanning_tree(G)

    return T


def construct_hclt_vtree(x, num_bins, sigma, chunk_size):
    mi = mutual_information_chunked(x, x, num_bins, sigma, chunk_size = chunk_size).detach().cpu().numpy()
    return chow_liu_tree(mi)
    

def HCLTGeneral(x: torch.Tensor, num_latents: int, 
         layer_fn: Type[CircuitNodes],
         num_bins: int = 32, 
         sigma: float = 0.5 / 32,
         chunk_size: int = 64,
         num_root_ns: int = 1,
         input_dist: Optional[Distribution] = None,
         input_node_type: Type[Distribution] = Categorical, 
         input_node_params: dict = {"num_cats": 256},
         input_params: torch.Tensor = None,
         block_size: int = None,
         permute_block_size: int = None,
         homogeneous_inputs: bool = True,
         homog_by_channel: bool = False,
         T = None,
         num_latents_schedule=None) -> CircuitNodes:
    """
    Construct Hidden Chow-Liu Trees (https://arxiv.org/pdf/2106.02264.pdf).

    :param x: the input data of size [# samples, # variables] used to construct the backbone Chow-Liu Tree
    :type x: torch.Tensor

    :param num_latents: size of the latent space
    :type num_latents: int

    :param num_bins: number of bins to divide the input data for mutual information estimation
    :type num_bins: int

    :param sigma: a variation parameter used when estimating mutual information
    :type sigma: float

    :param chunk_size: chunk size to compute mutual information (consider decreasing if running out of GPU memory)
    :type chunk_size: int

    :param num_root_ns: number of root nodes
    :type num_root_ns: int

    :param input_dist: input distribution
    :type input_dist: Distribution
    """

    if input_dist is not None:
        input_node_type, input_node_params = input_dist._get_constructor()
    
    if T is None:
        T = construct_hclt_vtree(x, num_bins, sigma, chunk_size)
    root = nx.center(T)[0]

    if block_size is not None:
        assert num_latents % block_size == 0, "num_latents must be divisible by sparse_block_size"
    else: # assume monarch
        assert (num_latents ** 0.5) == int(num_latents ** 0.5), "num_latents must be a square number"
        block_size = int(num_latents ** 0.5)

    root_r = BayesianTreeToHiddenRegionGraphGeneral(
        T, root, num_latents, input_node_type,
        input_node_params, layer_fn,
        num_root_ns = num_root_ns,
        block_size = block_size, permute_block_size=permute_block_size,
        homogeneous_inputs=homogeneous_inputs, input_params=input_params,
        homog_by_channel=homog_by_channel,
        num_latents_schedule=num_latents_schedule
    )
    
    return root_r