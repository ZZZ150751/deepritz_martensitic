import math
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import math


def W(u_x, u_y):
    return ((u_x ** 2.0) * ((1.0 - u_x) ** 2.0) + u_y ** 2.0) / 2

#激活函数
class SmReLU(nn.Module):
    def __init__(self, rho):
        super().__init__()
        self.rho = rho
    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x**2.0 + self.rho**2.0))

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
# 生成所有边界点
def boundary(params, device):
    N = params["b_batch_size"]

    # 左边界 (x=0, y ∈ [0,1])
    y_left = torch.rand(N, 1, device=device)
    x_left = torch.zeros(N, 1, device=device)
    points_left = torch.cat([x_left, y_left], dim=1)

    # 右边界 (x=L, y ∈ [0,1])
    y_right = torch.rand(N, 1, device=device)
    x_right = params["L"] * torch.ones(N, 1, device=device)
    points_right = torch.cat([x_right, y_right], dim=1)

    # 下边界 (y=0, x ∈ [0,L])
    x_bottom = torch.rand(N, 1, device=device) * params["L"]
    y_bottom = torch.zeros(N, 1, device=device)
    points_bottom = torch.cat([x_bottom, y_bottom], dim=1)

    # 上边界 (y=1, x ∈ [0,L])
    x_top = torch.rand(N, 1, device=device) * params["L"]
    y_top = torch.ones(N, 1, device=device)
    points_top = torch.cat([x_top, y_top], dim=1)

    # 合并所有边界点
    points = torch.cat([points_left, points_right, points_bottom, points_top], dim=0)

    return points  # (4N, 2)



#计算E的取点方法：网格取点，得到 （batch * k）x 2维张量
def grid(params, device):
    xr = torch.linspace(0, params["L"], params["L"] * params["batch_size"], device=device)#x向量
    yr = torch.linspace(0, 1, params["batch_size"], device=device)#y向量

    grid_x, grid_y = torch.meshgrid(xr, yr, indexing='ij')
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    return points, xr, yr


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
        xb = boundary(params, device)

        output_r = model(xr)
        output_b = model(xb) #(4 * b_batch_size, 1)维张量

        u_b = 0.5 * xb[:, 0:1]

        # 自动微分
        grad_u = torch.autograd.grad(outputs=output_r, inputs=xr,
                                     grad_outputs=torch.ones_like(output_r),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

        grad_x = grad_u[:, 0:1]
        grad_y = grad_u[:, 1:2]

        # 能量损失
        loss_e = torch.mean(W(grad_x, grad_y))

        # 边界损失
        loss_b = torch.mean((output_b - u_b) ** 2)

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
        "lr": 1e-3,  # 学习率
        "n": 5,  # 隐藏层数
        "L": 1, # x边界限制

        "gamma": 0.5,  # 边界条件参数
        "tau": 500.0,  # 边界惩罚系数
        #第一种方法取点：
        "i_batch_size": 1024, #内点
        "b_batch_size": 256, #边界
        "batch_size": 100,  #取点批次大小
        "num_epochs": 50000,  # 迭代次数
    }

    # DNN配置（隐藏层神经元个数）
    DNN_dict = {
        "DNN1": 16,
        "DNN2": 32,
        "DNN3": 64,
        "DNN4": 256
    }

    energy_list = []
    layers = []

    for name, m in DNN_dict.items():
        print(f"\nTraining with {name} layers.")
        params["m"] = m
        params["model_name"] = f"width_{m}"

        model = DNN(params["d"], params["m"], params["n"], SmReLU(rho = 0.1)).to(device)
        model.apply(DNN.init_trunc_xavier)
        trained_model = train(model, device, params)

        trained_model.load_state_dict(torch.load(f"model_{params['model_name']}.mdl"))
        trained_model.eval()


        points, xr_t, yr_t = grid(params, device)
        points.requires_grad_(True)

        u_pred = trained_model(points)

        grad_u = torch.autograd.grad(outputs=u_pred, inputs=points,
                                 grad_outputs=torch.ones_like(u_pred),
                                 create_graph=False, retain_graph=False,
                                 only_inputs=True)[0]

        # 取 u_x
        ux_pred = grad_u[:, 0:1].detach().cpu().numpy().reshape(params["batch_size"], params["batch_size"])

        grad_x = grad_u[:, 0:1]
        grad_y = grad_u[:, 1:2]

        integrand = W(grad_x, grad_y)
        hx = (xr_t[-1] - xr_t[0]).item() / (len(xr_t) - 1)
        hy = (yr_t[-1] - yr_t[0]).item() / (len(yr_t) - 1)

        energy = (hx * hy) * integrand.sum().item()
        energy_list.append(energy)

        h = math.log(m, 2)
        layers.append(h)

        xr = xr_t.cpu().numpy()
        yr = yr_t.cpu().numpy()
        X, Y = np.meshgrid(xr, yr, indexing='ij')


        plt.figure(figsize=(16, 8))
        plt.pcolormesh(X, Y, ux_pred, shading='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"u_x when γ={params['gamma']} ")

        plt.savefig(f'ux_width{m}.png', dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(layers, energy_list, marker='o',linewidth=2,
                    color="blue")
    plt.xlabel("log_2(W_NN)")
    plt.ylabel("E")
    plt.title("γ=0.5")
    plt.grid(True)
    plt.savefig("Fig9_a.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()