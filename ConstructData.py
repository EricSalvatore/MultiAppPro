import networkx as nx
import random
import math
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 训练种子
random.seed(66)
np.random.seed(66)


# 生成app属性
def generate_app_properties(num_apps, traffic_mean, traffic_std_dev, yield_rate_min, yield_rate_max, cost_fixed_min,
                            cost_fixed_max, cost_variable_ratio, categories, developer_types):
    # 每一个app有以下属性  返回的是{{id, cate, ...}， {id, cate, ...}， {id, cate, ...}}
    # id: app的id
    # category: app类别
    # developer_type: app开发者类型 个人和公司两种类型
    # traffic： 生成一个符合正态分布的流量数值
    # yield_rate：流量使用效率
    # cost: 花费 = 固定花费 + 非固定花费； 服务效率 = （流量 * 流量使用效率）- 流量花费

    apps = []
    for _ in range(num_apps):
        traffic = int(round(random.lognormvariate(traffic_mean, traffic_std_dev)))
        yield_rate = random.uniform(yield_rate_min, yield_rate_max)
        fixed_cost = random.uniform(cost_fixed_min, cost_fixed_max)
        variable_cost = cost_variable_ratio * traffic
        cost = fixed_cost + variable_cost
        service_efficiency = (yield_rate * traffic) - cost

        category = random.choice(categories)
        developer_type = random.choice(developer_types)

        apps.append({
            'id': f'App_{_}',
            'category': category,
            'developer_type': developer_type,
            'traffic': traffic,
            'yield_rate': yield_rate,
            'cost': cost,
            'service_efficiency': service_efficiency
        })
    return apps


def generate_node_features(apps, categorical_features, numerical_features):
    # 构建节点特征，特征有两种类型 一种是类别特征  一种是数值特征
    # 计算新特征矩阵的列数
    # categorical_features = ['category', 'developer_type']
    # numerical_features = ['traffic', 'yield_rate', 'cost']
    # 每一个数值特征处理一下 形成三种不同特征：归一化特征，平方，自然对数 三种处理
    #
    num_numerical_features = len(numerical_features) * 3  # 每个数值特征将有2个新特征（平方和对数）
    num_categorical_features = len(categorical_features)  # 类别特征保持不变
    num_features = num_numerical_features + num_categorical_features

    # 初始化特征矩阵 矩阵尺寸：app数目 * 特征数目
    node_features = np.zeros((len(apps), num_features))

    # 特征归一化
    scaler = StandardScaler()

    # 处理数值型属性
    feature_index = 0
    # ['traffic', 'yield_rate', 'cost'] ==> ['traffic 归一化', 'traffic 平方', 'traffic 对数'， 'yield_rate 归一化'， 'yield_rate 平方', 'yield_rate 对数', 'cost 归一化', 'cost 平方', 'cost 对数']
    for i, feature in enumerate(numerical_features):
        numerical_data = np.array([app[feature] for app in apps])

        # 归一化原始特征
        X_scaled = scaler.fit_transform(numerical_data.reshape(-1, 1))
        node_features[:, feature_index] = X_scaled.ravel()
        feature_index += 1

        # 构造新特征：平方
        squared_data = numerical_data ** 2
        node_features[:, feature_index] = squared_data
        feature_index += 1

        # 构造新特征：自然对数，避免对数为负
        logarithmic_data = np.log(numerical_data + 1)
        node_features[:, feature_index] = logarithmic_data
        feature_index += 1

    # 处理类别型属性
    for i, feature in enumerate(categorical_features): # ['category', 'developer_type']
        # 分别记录 len(apps) 个 category 和 developer_type类别
        category_apps = np.array([app[feature] for app in apps])

        # 标签编码
        label_encoder = {cat: idx for idx, cat in enumerate(np.unique(category_apps))}
        category_apps_encoded = np.array([label_encoder[cat] for cat in category_apps])
        node_features[:, feature_index] = category_apps_encoded
        feature_index += 1
    # 每一个app节点的特征为：
    # ['traffic 归一化', 'traffic 平方', 'traffic 对数'， 'yield_rate 归一化'， 'yield_rate 平方', 'yield_rate 对数', 'cost 归一化', 'cost 平方', 'cost 对数', 'category', 'developer_type']
    return node_features


def create_graph(apps):
    G = nx.Graph()
    for app in apps:
        G.add_node(app['id'], **app)
    return G


def build_static_network(G, apps, category_weights, developer_type_weights):
    # 生成静态网络
    for app in apps:
        for other_app in apps:
            if app['id'] != other_app['id']:
                # 避免除以零，如果流量相同，流量相似度为1
                traffic_similarity = (app['traffic'] != other_app['traffic']) * \
                                     (1 - (abs(app['traffic'] - other_app['traffic']) / max(abs(app['traffic']),
                                                                                            abs(other_app['traffic']),
                                                                                            1)))
                category_weight = category_weights.get(app['category'], 1)
                developer_type_weight = developer_type_weights.get(app['developer_type'], 1)

                # 联合权重=类别权重、开发类型权重、流量相似度
                combined_weight = category_weight * developer_type_weight * traffic_similarity

                # 对构建好的图 添加图的边 构建带权图
                G.add_edge(app['id'], other_app['id'], weight=combined_weight, rule='combined_similarity')


def construct_data_model(**kwargs):
    # 提取参数，设置默认值
    num_apps = kwargs.get('num_apps', 1000)
    traffic_mean = kwargs.get('traffic_mean', 2)
    traffic_std_dev = kwargs.get('traffic_std_dev', 1)
    yield_rate_min = kwargs.get('yield_rate_min', 0.05)
    yield_rate_max = kwargs.get('yield_rate_max', 0.15)
    cost_fixed_min = kwargs.get('cost_fixed_min', 100)
    cost_fixed_max = kwargs.get('cost_fixed_max', 500)
    cost_variable_ratio = kwargs.get('cost_variable_ratio', 0.1)

    categories = kwargs.get('categories', ['education', 'gaming', 'tools', 'social', 'health'])
    developer_types = kwargs.get('developer_types', ['individual', 'company'])

    category_weights = kwargs.get('category_weights', {
        'education': 1.2, 'gaming': 1.5, 'tools': 1.0, 'social': 1.8, 'health': 1.3
    })

    developer_type_weights = kwargs.get('developer_type_weights', {
        'individual': 0.8, 'company': 1.2
    })

    # 生成每一个app的属性集合
    apps = generate_app_properties(
        num_apps,
        traffic_mean,
        traffic_std_dev,
        yield_rate_min,
        yield_rate_max,
        cost_fixed_min,
        cost_fixed_max,
        cost_variable_ratio,
        categories,
        developer_types
    )

    # 以每一个app为节点 生成一个图
    G = create_graph(apps)
    # 使用初始化图、每一个app的属性信息、类别属性、开发者类型属性生成一个静态网络
    build_static_network(G, apps, category_weights, developer_type_weights)

    print(f"Generated graph with {len(apps)} apps.")
    print("Example node properties:", next(iter(G.nodes(data=True))))  # 修正语法错误
    print(f"Number of edges in the static network: {G.number_of_edges()}")
    #     print("All node properties:")
    #     for node, attributes in G.nodes(data=True):
    #         print(f"Node {node} has attributes: {attributes}")

    # 打印边的属性来展示不同构建规则和权重
    #     for u, v, attrs in G.edges(data=True):
    #         print(f"Edge between {u} and {v} was built based on rule: {attrs['rule']} with weight {attrs['weight']}")

    # 定义类别型和数值型属性
    categorical_features = ['category', 'developer_type']
    numerical_features = ['traffic', 'yield_rate', 'cost']

    # 假设apps是包含所有app属性的列表
    # 调用函数生成节点特征
    node_features = generate_node_features(apps, categorical_features, numerical_features)
    #     print(node_features)
    # 将特征赋给图G的节点
    for i, app in enumerate(apps):
        G.nodes[app['id']]['features'] = node_features[i]
    return G

if __name__ == "__main__":
    construct_data_model(num_apps=1500, traffic_mean=10, traffic_std_dev=0.4, yield_rate_min=0.04, yield_rate_max=0.14,
         cost_fixed_min=100, cost_fixed_max=500, cost_variable_ratio=0.02)