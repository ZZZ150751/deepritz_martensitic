import math
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam
import matplotlib.pyplot as plt


def W(z):
    return (z ** 2) * ((1.0 - z) ** 2)

class DNN(nn.Module):
    def __init__(self, d, m, n, activation): #参数：输入维数，隐藏层神经元数，隐藏层个数，激活函数
        super(DNN, self).__init__()

        #输入层
        self.input_layer = nn.Linear(d, m)
        #激活函数
        self.activation = activation

        #隐藏层
        self.hidden_stack = nn.ModuleList()
        for _ in range(n):
            self.hidden_stack.append(nn.Linear(m, m))

        #输出层
        self.output_layer = nn.Linear(m, 1)

    # 权重初始化的函数（截断正态分布，方差为：2/(dim_in + dim_out)）
    @staticmethod   #在类中
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


def train(model, device, params):
    print(model)
    print("总参数个数：", sum(p.numel() for p in model.parameters()))

    optimizer = Adam(model.parameters(), lr=params["lr"])
    model.train()

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, params["num_epochs"] + 1):
        xr = torch.rand(params["batch_size"], 1, requires_grad=True, device=device)

        output_r = model(xr)

        #自动微分
        grad_u = torch.autograd.grad(
            outputs=output_r,
            inputs=xr,
            grad_outputs=torch.ones_like(output_r),
            create_graph=True,
            retain_graph=True
        )[0]

        #能量损失
        loss_e = torch.mean(W(grad_u))

        #边界损失
        x_b0 = torch.zeros(1, 1, device=device)
        x_b1 = torch.ones(1, 1, device=device)

        output_b0 = model(x_b0)
        output_b1 = model(x_b1)
        loss_b = (output_b0 - 0.0) ** 2 + (output_b1 - params["gamma"]) ** 2

        #总损失
        loss = loss_e + params["tau"] * loss_b

        #反向传播
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
        "d": 1,  # 输入维度
        "lr": 1e-2,  #学习率

        "gamma": 0.5,  # 边界条件参数
        "tau": 500.0,   # 边界惩罚系数
        "batch_size": 512,   # 批次大小
        "num_epochs": 100000   # 迭代次数
    }

    #DNN配置（隐藏层神经元个数，隐藏层数）
    DNN_dict = {
        "DNN1": [64, 3],
        "DNN2": [128, 3],
        "DNN3": [64, 4],
        "DNN4": [128, 4],
        "DNN5": [64, 5],
        "DNN6": [128, 5],
    }

    plt.figure(figsize=(12, 8))

    #测试数据
    x_test = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)

    for name, size in DNN_dict.items():
        print(f"\nTraining with {name} .")
        params["m"] = size[0]
        params["n"] = size[1]
        params["model_name"] = f"{size[1]}x{size[0]}"

        model = DNN(params["d"], params["m"], params["n"], nn.ReLU()).to(device)

        # 权重与偏置初始化
        model.apply(DNN.init_trunc_xavier)

        trained_model = train(model, device, params)

        trained_model.load_state_dict(torch.load(f"model_{params['model_name']}.mdl"))

        trained_model.eval()

        with torch.no_grad():
            u_pred = trained_model(x_test).cpu().numpy()
            if name == "DNN1":
                plt.plot(
                    x_test.cpu().numpy(),
                    u_pred,
                    label= f"{size[1]} × {size[0]}",
                    linewidth=2,
                    color="red"
                )
            if name == "DNN2":
                plt.plot(
                    x_test.cpu().numpy(),
                    u_pred,
                    label=f"{size[1]} × {size[0]}",
                    linewidth=2,
                    color="lime",
                    linestyle="-."
                )
            if name == "DNN3":
                plt.plot(
                    x_test.cpu().numpy(),
                    u_pred,
                    label=f"{size[1]} × {size[0]}",
                    linewidth=2,
                    color="b",
                    linestyle="-."
                )
            if name == "DNN4":
                plt.plot(
                    x_test.cpu().numpy(),
                    u_pred,
                    label=f"{size[1]} × {size[0]}",
                    linewidth=2,
                    color="cyan",
                    linestyle="--"
                )

            if name == "DNN5":
                plt.plot(
                    x_test.cpu().numpy(),
                    u_pred,
                    label=f"{size[1]} × {size[0]}",
                    linewidth=2,
                    color="fuchsia",
                    linestyle="--"
                )

            if name == "DNN6":
                plt.plot(
                    x_test.cpu().numpy(),
                    u_pred,
                    label=f"{size[1]} × {size[0]}",
                    linewidth=2,
                    color="black",
                    linestyle="--"
                )


    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"γ={params['gamma']} ")
    plt.legend()
    plt.grid(True)
    plt.savefig('fig.5_b.png')
    plt.show()


if __name__ == "__main__":
    main()





