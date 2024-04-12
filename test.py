import plotly.graph_objects as go

# 测试数据
data = [
    (1, 5, 74.02, 2.96), 
    (5, 1, 26.04, 1.04),
    (2, 5, 76.80, 1.54),
    (5, 2, 45.80, 0.92),
    (3, 4, 64.38, 1.07),
    (3, 5, 76.07, 1.01),
    (4, 3, 57.37, 0.96),
    (4, 4, 71.84, 0.90),
    (4, 5, 84.73, 0.85),
    (5, 3, 62.69, 0.84),
    (5, 4, 97.35, 0.97)
]

servers = [x[0] for x in data]
models_per_server = [x[1] for x in data]
total_time = [x[2] for x in data]
avg_time = [x[3] for x in data]

# 创建散点图
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=servers,
    y=total_time,
    mode='markers',
    name='Total Time',
    marker=dict(
        size=10,
        color=total_time,
        colorscale='Viridis',
        showscale=True
    ),
    text=[f"Models per Server: {m}" for m in models_per_server],
    hoverinfo='text+x+y'
))

fig.add_trace(go.Scatter(
    x=servers,
    y=avg_time,
    mode='markers',
    name='Average Time',
    marker=dict(
        size=10,
        color=avg_time,
        colorscale='Plasma',
        showscale=True
    ),
    text=[f"Models per Server: {m}" for m in models_per_server],
    hoverinfo='text+x+y'
))

fig.update_layout(
    title='Performance Comparison',
    xaxis_title='Number of Servers',
    yaxis_title='Time (s)',
    hovermode='closest'
)

fig.show()

# 找出平均时间最短的配置
best_config = min(data, key=lambda x: x[3])
print(f"Best configuration: {best_config[0]} servers with {best_config[1]} models per server")
print(f"Average time: {best_config[3]}s")