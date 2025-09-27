import math
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def W(u_x, u_y):
    return ((u_x ** 2.0) * ((1.0 - u_x) ** 2.0) + u_y ** 2.0) / 2

class DNN(nn.Module):
    def __init__(self, d, m, n, activation):  # 参数：输入维数，隐藏层神经元数，隐藏层个数，激活函数
        super(DNN, self).__init__()

        # 输入层
        self.input_layer = nn.Linear(d, m)
        # 激活函数
        self.activation = activation

        # 隐藏层
        self.hidden_stack = nn.ModuleList()
        for _ in range(n):
            self.hidden_stack.append(nn.Linear(m, m))

        # 输出层
        self.output_layer = nn.Linear(m, 1)

    # 权重初始化的函数（截断正态分布，方差为：2/(dim_in + dim_out)）
    @staticmethod  # 在类中
    def init_trunc_xavier(module):
        if isinstance(module, nn.Linear):
            d_in = module.in_features
            d_out = module.out_features
            std = math.sqrt(2.0 / (d_in + d_out))
            init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)

        # 遍历所有隐藏层
        for layer in self.hidden_stack:
            x = layer(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x

#第一种取点方法：均匀取点，从内点和边界点分别均匀取点
#1.内点
def interior(params, device):
    points = torch.rand(params["i_batch_size"], 2, device=device)
    points[:, 0] = points[:, 0] * params["L"]
    points[:, 1] = points[:, 1]
    return points    # (i_batch_size, 2)维张量

# 2.边界点
def boundary_1(params, device):
    # 在 y ∈ [0,1] 上均匀采样
    y0 = torch.rand(params["b_batch_size"], 1, device=device)
    y1 = torch.rand(params["b_batch_size"], 1, device=device)

    # 在 x = 0 上均匀采样
    x0 = torch.zeros(params["b_batch_size"], 1, device=device)

    # 拼接边上的点
    points = torch.cat([x0, y0], dim=1)  # (b_batch_size, 2)维张量
    return points

def boundary_2(params, device):
    # 在 y ∈ [0,1] 上均匀采样
    y0 = torch.rand(params["b_batch_size"], 1, device=device)
    y1 = torch.rand(params["b_batch_size"], 1, device=device)

    # 在 x = L 上均匀采样
    xL = params["L"] * torch.ones(params["b_batch_size"], 1, device=device)

    # 拼接边上的点
    points = torch.cat([xL, y1], dim=1)  # (b_batch_size, 2)维张量
    return points


#第二种取点方法：网格取点，得到 （batch ** 2）x 2维张量
def grid(params, device):
    xr = torch.linspace(0, params["L"], params["batch_size"], device=device)
    yr = torch.linspace(0, 1, params["batch_size"], device=device)

    grid_x, grid_y = torch.meshgrid(xr, yr, indexing='ij')
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    return points


def train(model, device, params):
    print(model)
    print("总参数个数：", sum(p.numel() for p in model.parameters()))

    optimizer = Adam(model.parameters(), lr=params["lr"])
    model.train()

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, params["num_epochs"] + 1):
        xr = interior(params, device)
        xr.requires_grad_(True)
        xb_1 = boundary_1(params, device)
        xb_2 = boundary_2(params, device)

        output_r = model(xr)
        output_b_1 = model(xb_1) #(b_batch_size, 1)维张量
        output_b_2 = model(xb_2)  #(b_batch_size, 1)维张量

        # 自动微分
        grad_u = torch.autograd.grad(outputs=output_r, inputs=xr,
                                     grad_outputs=torch.ones_like(output_r),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

        grad_x = grad_u[:, 0:1]
        grad_y = grad_u[:, 1:2]

        # 能量损失
        loss_e = torch.mean(W(grad_x, grad_y))

        # 边界损失
        loss_b = torch.mean(output_b_1 ** 2.0) + torch.mean((output_b_2 - params["gamma"]) ** 2.0)

        # 总损失
        loss = loss_e + params["tau"] * loss_b

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            torch.save(model.state_dict(), f"model_{params['model_name']}.mdl")

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}/{params["num_epochs"]}, Loss: {loss.item():.6f}, '
                  f'Energy Loss: {loss_e.item():.6f}, Boundary Loss: {loss_b.item():.6f}')

    print(f'LR={params["lr"]}, Best epoch={best_epoch}, Best loss={best_loss:.6f}')

    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    params = {
        "d": 2,  # 输入维度
        "lr": 1e-2,  # 学习率
        "m": 128,  # 隐藏层神经元个数
        "n": 3,  # 隐藏层数
        "L": 1, # x边界限制

        "gamma": 0.25,  # 边界条件参数
        "tau": 500.0,  # 边界惩罚系数
        #第一种方法取点：
        "i_batch_size": 1024, #内点
        "b_batch_size": 512, #边界
        "batch_size": 100,  # 第二种方法取点批次大小
        "num_epochs": 100000,  # 迭代次数
        "model_name": "deepritz_fig7_a1"
    }

    model = DNN(params["d"], params["m"], params["n"], nn.ReLU()).to(device)
    model.apply(DNN.init_trunc_xavier)
    trained_model = train(model, device, params)

    trained_model.load_state_dict(torch.load(f"model_{params['model_name']}.mdl"))
    trained_model.eval()

    # 创建网格数据用于绘图
    nx, ny = 100, 100
    x = np.linspace(0, params["L"], nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 转换为张量并预测
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1),
                          dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = trained_model(points).cpu().numpy()

    # 重塑预测结果
    U_pred = u_pred.reshape(nx, ny)

    # 画 u 的 3D 表面图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(Y, X, U_pred, cmap=cm.Wistia, edgecolor='k', linewidth=0.2, alpha=0.8)
    ax.view_init(azim=225)

    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    ax.set_title(f"γ={params['gamma']} ")

    plt.savefig('fig7_a1.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()