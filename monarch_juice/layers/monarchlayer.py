import torch
from pyjuice.nodes import multiply, summate


def create_dense_layer(*input_nodes, num_node_blocks, block_size, homogeneous, existing_nodes=None, num_layers=1, **kwargs):
    sum_nodes = dict()
    np = multiply(*input_nodes)
    for i in range(num_layers):
        if existing_nodes is None:
            ns = summate(np, num_node_blocks = num_node_blocks, block_size=block_size)
            sum_nodes[f'ns{i}'] = ns
        else:
            ns = existing_nodes[f'ns{i}'].duplicate(np, tie_params = homogeneous)
            sum_nodes[f'ns{i}'] = existing_nodes[f'ns{i}']
        
        if i < num_layers - 1:
            np = multiply(ns, edge_ids = torch.arange(0, num_node_blocks)[:, None]) # passthrough layer
        
    return ns, sum_nodes


def _create_monarch_layer(ns, num_node_blocks, block_size, permute_block_size, existing_node, homogeneous):
    permuted_edges = torch.arange(0, block_size * num_node_blocks).reshape(
            num_node_blocks * block_size // permute_block_size, permute_block_size
        ).permute(1, 0).reshape(
            block_size * num_node_blocks
        )[:,None]
    np = multiply(ns, edge_ids = permuted_edges, sparse_edges = True)

    edge_ids = torch.arange(0, num_node_blocks)[None,:].repeat(2, 1) # Tensor([[0, 1, ...], [0, 1, ...]])
    if existing_node is None:
        return summate(np, edge_ids = edge_ids, block_size=block_size)
    else:
        return existing_node.duplicate(np, tie_params = homogeneous)
    

def create_monarch_layers(*input_nodes, num_node_blocks, block_size, homogeneous, existing_nodes=None,
                          permute_block_size=None, num_layers=3, **kwargs):
    """For num_layers=3, implements a composition of two Monarch matrices.
    """
    sum_nodes = dict()
    if permute_block_size is None:
        permute_block_size = block_size
    else:
        assert permute_block_size % block_size == 0, "permute_block_size must be divisible by block_size"

    np0 = multiply(*input_nodes)

    # Create a block-diagonal sum layer
    edge_ids = torch.arange(0, num_node_blocks)[None,:].repeat(2, 1) # Tensor([[0, 1, ...], [0, 1, ...]])
    if existing_nodes is None:
        ns = summate(np0, edge_ids = edge_ids, block_size=block_size)
        sum_nodes['ns1'] = ns
    else:
        ns = existing_nodes['ns1'].duplicate(np0, tie_params = homogeneous)

    for layer_num in range(2, num_layers + 1):
        ns = _create_monarch_layer(ns, num_node_blocks, block_size, permute_block_size, existing_nodes.get(f'ns{layer_num}', None) if existing_nodes is not None else None, homogeneous)
        sum_nodes[f'ns{layer_num}'] = ns

    sum_nodes = existing_nodes if existing_nodes is not None else sum_nodes
    return ns, sum_nodes