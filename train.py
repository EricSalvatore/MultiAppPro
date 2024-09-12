import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv, SAGEConv, GCNConv
from torch_geometric.data import HeteroData
import numpy as np
from ConstructData import *

num_apps = 150
# 通过上一步构建的当前时间节点的图网络结构
G = construct_data_model(num_apps=150, traffic_mean=10, traffic_std_dev=0.4, yield_rate_min=0.04, yield_rate_max=0.14,
                         cost_fixed_min=100, cost_fixed_max=500, cost_variable_ratio=0.02)
# m = G.nodes.get('App_0')
# 2024-09-11 dhj
# 添加categories节点类别序列
categories = ['education', 'gaming', 'tools', 'social', 'health']
# 所有节点的流量使用效率map
nodes_yield_rate_map = {node: G.nodes[node]['yield_rate'] for node in G.nodes}
# 所有节点的流量map
nodes_traffic_map = {node: G.nodes[node]['traffic'] for node in G.nodes}
# 所有节点的花费map
nodes_cost_map = {node: G.nodes[node]['cost'] for node in G.nodes}
# 2024-09-11 dhj
# 构建类别索引到node的映射
idx_node_map = {}


# 2024-09-11 dhj
# 构建异构数据 这里构建异构数据是通过节点类型来进行的
def create_hetero_data(G, categories):
    """
    :param G: 当前时间切片下的图结构
    :param categories: 节点类型节点的类型list
    :return:
    """
    # 构建节点和边的字典
    hetero_data = HeteroData()
    # 节点到索引的映射
    node_idx_map = {}
    # 构建异构节点数据
    for category in categories:
        node_indices = [node for node in G.nodes if G.nodes[node]['category'] == category]
        app_features = [G.nodes[node]['features'] for node in node_indices]
        hetero_data[category].x = torch.tensor(app_features, dtype=torch.float)

        # 为每种类别的节点创建一个节点到索引的映射
        node_idx_map[category] = {node: i for i, node in enumerate(node_indices)}
        idx_node_map[category] = {i: node for i, node in enumerate(node_indices)}

    # 为异构数据添加边以及权重，连接相同类别的节点
    for category in categories:
        edge_index = []
        edge_weight = []
        hetero_data[category, 'connected_to', category].edge_index = torch.empty((2, 0), dtype=torch.long)
        hetero_data[category, 'connected_to', category].edge_weight = torch.empty((0,), dtype=torch.float)

        for u, v, data in G.edges(data=True):
            if G.nodes[u]['category'] == category and G.nodes[v]['category'] == category:
                edge_index.append([node_idx_map[category][u], node_idx_map[category][v]])
                edge_weight.append(int(data['weight']))

        # 检查空边
        if len(edge_index) > 0:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            hetero_data[category, 'connected_to', category].edge_index = edge_index_tensor

        if len(edge_weight) > 0:
            edge_weight_tensor = torch.tensor(edge_weight, dtype=torch.float)
            hetero_data[category, 'connected_to', category].edge_weight = edge_weight_tensor

    return hetero_data


hetero_data = create_hetero_data(G, categories)


# 定义一个两层的异构图神经网络 来获取新的节点表征
class ReconstructorHGNN(nn.Module):
    def __init__(self, meta_data, hidden_dim, out_dim):
        super(ReconstructorHGNN, self).__init__()
        self.meta_data = meta_data
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        conv1_map = {}
        conv2_map = {}

        # 获取实际存在的连接类型
        existing_edge_types = meta_data[1]

        for connection in existing_edge_types:
            conv1_map[connection] = GCNConv(-1, hidden_dim)
            conv2_map[connection] = GCNConv(-1, out_dim)

        self.conv1 = HeteroConv(conv1_map, aggr='sum')
        self.conv2 = HeteroConv(conv2_map, aggr='sum')

    def forward(self, hetero_data):
        x_dict = hetero_data.x_dict
        edge_index_dict = hetero_data.edge_index_dict
        edge_weight_dict = hetero_data.edge_weight_dict

        x_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict, edge_weight_dict)
        # 20240911 dhj 通过节点类型异构网络 生成新的节点表示，dict{5}--> dict{'education': shape(29, 11), 'gaming': shape(31, 11), 'tools': shape(28, 11), 'social': shape(30, 11), 'health': shape(32, 11)}
        return x_dict


# 20240912 dhj 利用新的节点生成全连接图
class Generate_Full_Connect_Graph:
    def __init__(self, out_dim):
        # 所有节点的节点index 和 节点名称的映射
        self.all_index_node_map = {}
        # 所有节点的节点名称 和 节点index的映射
        self.all_node_index_map = {}
        # 所有节点
        self.all_node = []
        # 所有特征节点
        self.all_feature_node = torch.empty((0, out_dim), dtype=torch.float)

    def cat_all_node(self, x_dict):
        # 拼接生成所有节点, 更新所有节点index和节点名称的双向映射
        all_node = []
        all_index = 0
        for category in categories:
            x_category = x_dict[category]
            self.all_feature_node = torch.cat([self.all_feature_node, x_category], dim=0)
            for index in range(x_category.shape[0]):
                self.all_node_index_map[idx_node_map[category][index]] = all_index
                self.all_index_node_map[all_index] = idx_node_map[category][index]
                all_node.append(all_index)
                all_index += 1
        return torch.tensor(all_node)

    def generate_full_connect_graph(self, x_dict):
        # 生成全连接的边
        self.all_node = self.cat_all_node(x_dict)
        # print(f"all_node.shape[0] is {self.all_node.shape[0]}")
        all_edge_index = []
        for u in range(self.all_node.shape[0]):
            # print(f"u = {u}, src Node {self.all_index_node_map[u]}")
            for v in range(u + 1, self.all_node.shape[0]):
                # print(f"v = {v}, dst Node {self.all_index_node_map[v]}")
                all_edge_index.append([u, v])

        # 转换all_edge_index向量
        all_edge_index = torch.tensor(all_edge_index, dtype=torch.long).t().contiguous()
        return all_edge_index


# 2024-09-11 dhj 构建一个线性注意力网络 初始化/调整权重参数
class LinearAttention(nn.Module):
    def __init__(self, in_dim):
        super(LinearAttention, self).__init__()
        self.attention = nn.Linear(2 * in_dim, 1)

    def forward(self, x_list, edge_index):
        edge_weight = []
        num_edge = edge_index.shape[1]
        for i in range(num_edge):
            # 获取每条边的端点的index
            u_index = edge_index[0][i]
            v_index = edge_index[1][i]
            # 获取每条边的两个端点的特征
            u_feature = x_list[u_index]
            v_feature = x_list[v_index]

            cat_feature = torch.cat([u_feature, v_feature], dim=-1)
            weight_logits = self.attention(cat_feature)
            edge_weight.append(weight_logits)
        # 遍历 edge_weight 中的每一个 tensor，进行标准化
        cnty = 0
        cat_edge_weight = torch.cat(edge_weight, dim=0).type(torch.float)
        normalized_edge_weight = (cat_edge_weight - torch.mean(cat_edge_weight, dim=0)) / torch.std(cat_edge_weight,
                                                                                                    dim=0)
        edge_weight_out = torch.sigmoid(normalized_edge_weight)
        return edge_weight_out

# 2024-09-12 dhj 设计一个新的图网络，进行流量传输操作，最大化网络服务效能
class MessagePassingNet(nn.Module):
    def __init__(self, n):
        """
        :param n: 惩罚系数
        """
        super(MessagePassingNet, self).__init__()
        self.n = n

    def forward(self, _edge_index, _edge_weight, _nodes_yield_rate_map, _nodes_traffic_map, _nodes_cost_map,
                _all_index_node_map=None, _all_node_index_map=None):
        self.all_index_node_map = _all_index_node_map
        self.all_node_index_map = _all_node_index_map

        # 转换 traffic, yield_rate, 和 cost 到 Tensor 形式，保持计算图
        nodes_yield_rate_tensor = torch.tensor([_nodes_yield_rate_map[node] for node in _all_index_node_map.values()],
                                               dtype=torch.float32, requires_grad=True)
        nodes_traffic_tensor = torch.tensor([_nodes_traffic_map[node] for node in _all_index_node_map.values()],
                                            dtype=torch.float32, requires_grad=True)
        nodes_cost_tensor = torch.tensor([_nodes_cost_map[node] for node in _all_index_node_map.values()],
                                         dtype=torch.float32, requires_grad=True)

        # 进行消息传递
        new_nodes_traffic_tensor = self.message_passing_network(_edge_index, _edge_weight, nodes_traffic_tensor)

        # 计算新的总服务效能
        total_service_efficiency = self.cal_service_efficiency(nodes_yield_rate_tensor, new_nodes_traffic_tensor,
                                                               nodes_cost_tensor)
        return new_nodes_traffic_tensor, total_service_efficiency

    def message_passing_network(self, _edge_index, _edge_weight, _nodes_traffic_tensor):
        # 创建一个副本来存储更新后的流量，防止在计算过程中直接修改原始流量
        new_nodes_traffic_tensor = _nodes_traffic_tensor.clone()

        edge_num = _edge_index.shape[1]

        # 遍历所有的边，进行消息传递
        for i in range(edge_num):
            src = _edge_index[0][i]
            dst = _edge_index[1][i]
            weight = _edge_weight[i]

            # 获取src和dst的流量（使用原始流量进行计算，而不是直接修改的流量）
            src_traffic = _nodes_traffic_tensor[src]
            dst_traffic = _nodes_traffic_tensor[dst]

            # 计算流量差
            traffic_diff = src_traffic - dst_traffic

            # 根据流量差传递流量，更新暂存的流量张量
            if traffic_diff > 0:
                transfer = traffic_diff * self.n * weight
                new_nodes_traffic_tensor[src] = new_nodes_traffic_tensor[src] - transfer
                new_nodes_traffic_tensor[dst] = new_nodes_traffic_tensor[dst] + transfer
            elif traffic_diff < 0:
                transfer = traffic_diff * self.n * weight
                new_nodes_traffic_tensor[src] = new_nodes_traffic_tensor[src] + transfer
                new_nodes_traffic_tensor[dst] = new_nodes_traffic_tensor[dst] - transfer

        # 返回新的流量张量
        return new_nodes_traffic_tensor

    def cal_service_efficiency(self, _nodes_yield_rate_tensor, _nodes_traffic_tensor, _nodes_cost_tensor):
        # 计算服务效率：yield_rate * traffic - cost
        total_efficiency = torch.sum(_nodes_yield_rate_tensor * _nodes_traffic_tensor - _nodes_cost_tensor)
        return total_efficiency


class MultiAppGraphNet(nn.Module):
    def __init__(self, _hetero_data, hidden_dim, out_dim, threshold=0.3, n=0.2):
        super(MultiAppGraphNet, self).__init__()
        self.hetero_data = _hetero_data
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # 阈值
        self.threshold = threshold
        # 惩罚系数
        self.n = n
        # 构建节点特征的异构网络
        self.reconstructor_model = ReconstructorHGNN(hetero_data.metadata(), hidden_dim=self.hidden_dim,
                                                     out_dim=self.out_dim)

        # 构建全连接图网络
        self.gfc = Generate_Full_Connect_Graph(out_dim=out_dim)

        # 权重线性注意力获取
        self.linear_attention = LinearAttention(out_dim)

        # 消息传递网络
        # self.message_passing_net = MessagePassingNet(n=self.n)

    def filter_threshold_for_edge(self, _edge_index, _edge_weight):
        # 根据阈值 筛选出需要保留的边
        mask = _edge_weight > self.threshold
        return _edge_index[:, mask], _edge_weight[mask]

    def forward(self, _hetero_data):
        # 异构网络构建节点特征
        new_x_dict = self.reconstructor_model(_hetero_data)
        # 构建全连接图
        edge_index = self.gfc.generate_full_connect_graph(new_x_dict)
        all_feature_node = self.gfc.all_feature_node
        ## 所有节点的index到节点名的映射
        all_index_node_map = self.gfc.all_index_node_map
        ## 所有节点的节点名到index的映射
        all_node_index_map = self.gfc.all_node_index_map

        # 线性注意力获取权重
        edge_weight = self.linear_attention(all_feature_node, edge_index)

        # 根据阈值筛选 获取流量传输网络的边 和 权重
        new_edge_index, new_edge_weight = self.filter_threshold_for_edge(_edge_index=edge_index,
                                                                         _edge_weight=edge_weight)

        # 消息传递过程，计算新的流量图和总效能
        # new_nodes_traffic_tensor, total_service_efficiency = self.message_passing_net(_edge_index=new_edge_index,
        #                                                                               _edge_weight=new_edge_weight,
        #                                                                               _nodes_yield_rate_map=nodes_yield_rate_map,
        #                                                                               _nodes_traffic_map=nodes_traffic_map,
        #                                                                               _nodes_cost_map=nodes_cost_map,
        #                                                                               _all_index_node_map=all_index_node_map,
        #                                                                               _all_node_index_map=all_node_index_map)

        # return total_service_efficiency
        return edge_index.sum()

# torch.autograd.set_detect_anomaly(True)

model = MultiAppGraphNet(_hetero_data=hetero_data, hidden_dim=11, out_dim=11)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()

    # 调用模型的前向传播，得到效率的tensor
    total_efficiency = model(hetero_data)

    # 损失函数：最大化效能，即最小化负效能
    loss = -total_efficiency

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Efficiency: {total_efficiency.item()}')
