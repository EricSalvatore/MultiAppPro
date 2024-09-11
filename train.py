import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv, SAGEConv, GCNConv
from torch_geometric.data import HeteroData
import numpy as np
from ConstructData import *

# 节点类型
NODE_TYPES = ['developer_type', 'category', 'app']

# 边类型
EDGE_TYPES = [('developer_type', 'develops', 'app'), ('app', 'belongs_to', 'category')]

categories_list = ['education', 'gaming', 'tools', 'social', 'health']
categories_map = dict(zip(categories_list, range(len(categories_list))))
developer_types_list = ['individual', 'company']
developer_types_map = dict(zip(developer_types_list, range(len(developer_types_list))))

num_apps = 150
# 通过上一步构建的当前时间节点的图网络结构
G = construct_data_model(num_apps=150, traffic_mean=10, traffic_std_dev=0.4, yield_rate_min=0.04, yield_rate_max=0.14,
                         cost_fixed_min=100, cost_fixed_max=500, cost_variable_ratio=0.02)
# m = G.nodes.get('App_0')
nodes_features_list = [G.nodes.get('App_' + str(item))['features'].tolist() for item in range(0, 150)]

# 2024-09-11 dhj
# 添加categories节点类别序列
categories = ['education', 'gaming', 'tools', 'social', 'health']
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
        print(node_indices)
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

        print("\nChecking node features:")
        for node_type, x in x_dict.items():
            print(f"Node Type: {node_type}, Feature shape: {x.shape}")

        print("\nChecking edge indices:")
        for edge_type, edge_index in edge_index_dict.items():
            print(f"Edge Type: {edge_type}, Edge Index shape: {edge_index.shape}")

        print("\nChecking edge weights:")
        for edge_type, edge_weight in edge_weight_dict.items():
            if edge_weight is not None:
                print(f"Edge Type: {edge_type}, Edge Weight shape: {edge_weight.shape}")
            else:
                print(f"Edge Type: {edge_type}, No edge weight provided.")

        x_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict, edge_weight_dict)
        # 20240911 dhj 通过节点类型异构网络 生成新的节点表示，dict{5}--> dict{'education': shape(29, 11), 'gaming': shape(31, 11), 'tools': shape(28, 11), 'social': shape(30, 11), 'health': shape(32, 11)}
        return x_dict


model = ReconstructorHGNN(hetero_data.metadata(), hidden_dim=11, out_dim=11)

m = model(hetero_data)
data = HeteroData()

# 类别节点
categories_data = torch.randn(5, 10)
data['category'].x = categories_data
# 开发者节点
developer_type_data = torch.randn(2, 10)
data['developer_type'].x = developer_type_data
# app节点
app_data = torch.tensor(np.array(nodes_features_list), dtype=torch.float32)

# 异构边 edge_index
nodes_list = [G.nodes.get("App_" + str(item)) for item in range(num_apps)]

# 获取节点流量
'''
这里已经拿到了当前时间切片的节点流量
'''
traffic_list = [node['traffic'] for node in nodes_list]
traffic = torch.tensor(traffic_list)

nodes_categories_list = [int(categories_map[node["category"]]) for node in nodes_list]
nodes_app_list = [int(node["id"].split('_')[1]) for node in nodes_list]
nodes_developer_type_list = [int(developer_types_map[node["developer_type"]]) for node in nodes_list]
# 构造开发者到应用的边
developer_to_app_type_edge = torch.tensor(np.array([nodes_developer_type_list, nodes_app_list]), dtype=torch.long)
# 构造应用到类别的边
app_to_category_edge = torch.tensor(np.array([nodes_app_list, nodes_categories_list]), dtype=torch.long)

# 类别权重字典
category_weight_dict = {
    'education': 1.2, 'gaming': 1.5, 'tools': 1.0, 'social': 1.8, 'health': 1.3
}
# 开发者权重字典
developer_type_weights_dict = {
    'individual': 0.8, 'company': 1.2
}
# 开发者到应用的边权重
edge_weight_developer_to_app = torch.tensor(
    np.array([developer_type_weights_dict[node["developer_type"]] for node in nodes_list]), dtype=torch.float32)
# 应用到类别的边权重
edge_weight_app_to_category = torch.tensor(
    np.array([category_weight_dict[node["category"]] for node in nodes_list]), dtype=torch.float32)

data["app", "belongs_to", "category"].edge_index = app_to_category_edge
data["developer_type", "develops", "app"].edge_index = developer_to_app_type_edge


class MyMessagePassing(MessagePassing):
    def __init__(self, node_types, edge_types, in_channels, out_channels):
        super(MyMessagePassing, self).__init__(aggr='add')
        self.node_types = node_types
        self.edge_types = edge_types
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear_dict = nn.ModuleDict({
            node_type: nn.Linear(in_channels.get(node_type, 0), out_channels.get(node_type, 0))
            for node_type in self.node_types
        })

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        out = {ntype: torch.zeros(x_dict[ntype].size(0), self.out_channels[ntype], device=x_dict[ntype].device,
                                  dtype=torch.float)
               for ntype in self.node_types}

        for (src_type, relation_ship, dest_type), edge_index in edge_index_dict.items():
            weight = edge_weight_dict[(src_type, relation_ship, dest_type)]
            src_x = x_dict[src_type]
            if src_type in self.in_channels and src_type in self.out_channels:
                src_x = self.linear_dict[src_type](src_x)
                src_x = torch.relu(src_x)
            mid_message = self.propagate(edge_index=edge_index,
                                         size=(x_dict[src_type].size(0), x_dict[dest_type].size(0)),
                                         x=src_x, weight=weight.view(-1, 1))
            out[dest_type] += mid_message
        return out

    def message(self, x_j, weight):
        if weight is not None:
            return x_j * weight
        return x_j


in_channels = {'developer_type': 10, 'category': 10, 'app': 11}
out_channels = {'developer_type': 11, 'category': 11, 'app': 11}
hidden_channels = {'developer_type': 15, 'category': 15, 'app': 15}


class MyHGNN(nn.Module):
    def __init__(self, node_types, edge_types, in_channels, hidden_channels, out_channels):
        super(MyHGNN, self).__init__()
        self.layer1 = MyMessagePassing(node_types, edge_types, in_channels, hidden_channels)
        self.layer2 = MyMessagePassing(node_types, edge_types, hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        x_dict = self.layer1(x_dict, edge_index_dict, edge_weight_dict)
        x_dict = self.layer2(x_dict, edge_index_dict, edge_weight_dict)
        return x_dict


x_dict = {'developer_type': developer_type_data, 'category': categories_data, 'app': app_data}
edge_index_dict = {
    ('developer_type', 'develops', 'app'): developer_to_app_type_edge,
    ('app', 'belongs_to', 'category'): app_to_category_edge
}
edge_weight_dict = {
    ('developer_type', 'develops', 'app'): edge_weight_developer_to_app,
    ('app', 'belongs_to', 'category'): edge_weight_app_to_category
}

model = MyHGNN(NODE_TYPES, EDGE_TYPES, in_channels, hidden_channels, out_channels)
output_dict = model(x_dict, edge_index_dict, edge_weight_dict)

#1.3 通过异构图卷积聚合得到的新的节点表示
num_apps = output_dict['app'].size(0)
#2.1 构建全连接网络
full_connection_edge_index = torch.tensor([[i, j] for i in range(num_apps) for j in range(i + 1, num_apps)],
                                          dtype=torch.long).t().contiguous()
full_connection_edge_weight = torch.ones(num_apps * (num_apps - 1) // 2, dtype=torch.float32)


# 2.2 全连接网络的注意力层
class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionLayer, self).__init__()
        self.scale = in_channels ** -0.5
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_norm = nn.LayerNorm(self.in_channels)
        self.linear = nn.Linear(self.in_channels, self.out_channels * 3, bias=False)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        q, k, v = torch.chunk(x, 3, dim=1)
        q = q * self.scale
        sim = torch.einsum("...i d, ...j d->...i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = F.softmax(sim, dim=-1)
        out = torch.einsum("...i j, ...j d->...i d", attn, v)
        return out


class AttentionNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionNetwork, self).__init__()
        self.attention = AttentionLayer(in_channels, out_channels)

    def forward(self, x):
        x = self.attention(x)
        return x


attention_network = AttentionNetwork(in_channels=11, out_channels=11)


# 更新边的权重
def update_edge_weight(edge_index, app_data, edge_weight, traffic, threshold=0.1, alpha=0.5):
    # 节点的注意力权重
    attn_weights = attention_network(app_data)
    # 边的权重==边的两个端点节点的注意力权重的乘积
    edge_weight = torch.sum(attn_weights[edge_index[0]] * attn_weights[edge_index[1]], dim=1)
    # 计算一条边两端点节点的流量差异值，并计算绝对值
    traffic_diff = torch.abs(traffic[edge_index[0]] - traffic[edge_index[1]])
    mask = traffic_diff > threshold
    # 根据流量更新边和权重
    new_edge_index = edge_index[:, mask]
    new_edge_weight = edge_weight[mask] * alpha
    return new_edge_index, new_edge_weight


# 2.3 得到新的数据共享网络
new_edge_index, new_edge_weight = update_edge_weight(edge_index=full_connection_edge_index,
                                                     edge_weight=full_connection_edge_weight,
                                                     app_data=output_dict['app'],
                                                     traffic=traffic)


# todo:流量更新的规则算法
class MyNewMessagePassing(MessagePassing):
    def __init__(self, node_types, edge_types, hidden_channels, out_channels):
        super(MyNewMessagePassing, self).__init__(aggr='add')
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.linears = nn.ModuleDict({
            node_type: nn.Linear(hidden_channels.get(node_type, 0), out_channels.get(node_type, 0))
            for node_type in self.node_types
        })

    def forward(self, x_dict, edge_index_dict: dict, edge_weight_dict: dict):
        # 初始化输出字典
        out = {ntype: torch.zeros(x.size(0), self.out_channels.get(ntype, 0), device=x.device)
               for ntype, x in x_dict.items()}
        # 遍历边
        for (src_type, relation_ship, dest_type), edge_index in edge_index_dict.items():
            weight = edge_weight_dict.get((src_type, relation_ship, dest_type), None)
            src_x = x_dict[src_type]
            src_x = self.linears[src_type](src_x)
            mid_message = self.propagate(edge_index, size=(src_x.size(0), src_x.size(0)),
                                         x=src_x, weight=weight.view(-1, 1))
            out[dest_type] += mid_message

        return out

    def message(self, x_j, weight):
        # todo:传输规则
        # chazhi =
        return x_j if weight is None else x_j * weight


class MyNewGNN(nn.Module):
    def __init__(self, node_types, edge_types, in_channels, hidden_channels, out_channels):
        super(MyNewGNN, self).__init__()
        self.layer1 = MyNewMessagePassing(node_types, edge_types, in_channels, hidden_channels)
        self.layer2 = MyNewMessagePassing(node_types, edge_types, hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        x_dict = self.layer1(x_dict, edge_index_dict, edge_weight_dict)
        x_dict = self.layer2(x_dict, edge_index_dict, edge_weight_dict)
        return x_dict


New_NODE_TYPES = ["app"]
New_EDGE_TYPES = [('app', 'connects', 'app')]
new_in_channels = {'app': 11}
new_out_channels = {'app': 11}
new_hidden_channels = {'app': 11}

msg_passing_model = MyNewGNN(New_NODE_TYPES, New_EDGE_TYPES, new_in_channels, new_hidden_channels, new_out_channels)
# reshape_new_edge_index = new_edge_index.transpose(1, 0)
reshape_new_edge_index = new_edge_index

new_app_edge_index_dict = {
    ('app', 'connects', 'app'): reshape_new_edge_index
}
new_app_edge_weight_dict = {
    ('app', 'connects', 'app'): new_edge_weight
}
new_x_dict = {'app': output_dict['app']}
msg_output_dict = msg_passing_model(new_x_dict, new_app_edge_index_dict, new_app_edge_weight_dict)

print(msg_output_dict)


# 定义损失函数
def compute_loss(output_dict, traffic, cost):
    # 损失函数：考虑产出率和成本
    yield_rate = torch.mean(output_dict['app'], dim=1)
    service_efficiency = yield_rate * traffic - cost
    loss = -torch.mean(service_efficiency)
    return loss, torch.mean(service_efficiency)


# 模型、优化器
model = MyNewGNN(New_NODE_TYPES, New_EDGE_TYPES, new_in_channels, new_hidden_channels, new_out_channels)
optimizer = optim.Adam(model.parameters(), lr=0.1)
cost = torch.tensor([G.nodes[f'App_{i}']['cost'] for i in range(num_apps)], dtype=torch.float32)

# 训练循环
for epoch in range(100):  # 示例训练100个epoch
    model.train()
    optimizer.zero_grad()
    msg_output_dict = model(new_x_dict, new_app_edge_index_dict, new_app_edge_weight_dict)
    loss, service_efficiency = compute_loss(msg_output_dict, traffic, cost)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}, 系统服务效能: {service_efficiency.item()}")
