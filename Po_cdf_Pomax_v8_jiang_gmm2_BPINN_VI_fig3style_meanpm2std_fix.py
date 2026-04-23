# -*- coding: utf-8 -*-
"""
Po_cdf_Pomax_v8_jiang_exactmodel_gmm2_BPINN_VI.py

在你现有 v7（GMM2 + NLL + mean-only 物理损失）基础上，把“网络参数 θ”做成随机变量，
用变分推断 (Variational Inference, VI) 做一个轻量版 B-PINN（更准确说：Bayesian Neural Net + PINN penalty）。

核心改动（对应你 v7 程序）：
1) 把 GMM2Net 的每个 nn.Linear 换成 BayesLinear：
   q(W)=N(mu_W, sigma_W^2),  q(b)=N(mu_b, sigma_b^2)
   前向传播时用重参数技巧 W = mu + sigma * eps 采样（训练/评估都可采样）
2) 训练目标从
      loss = NLL + lam_phys * physics_loss
   变成 ELBO（负的证据下界）形式：
      loss = NLL + lam_phys * physics_loss + beta * KL(q(θ)||p(θ)) / N_data
   其中 p(θ) 是权重先验（默认 N(0, prior_sigma^2)）
3) 评估时，用多个 θ 采样得到一组 CDF：
   - 画“后验预测均值 CDF”
   - 可选画 95% 置信带（epistemic 不确定性）
   - 仍计算 ARMS（用预测均值 CDF vs MC-CDF）

保持不变：
- 姜华 3 节点设定（不确定性/约束/用 Gurobi 解 Pomax 的标签生成）
- mean-only 物理损失定义（用混合均值 mu_mix 检查电压/线路/机组约束）
"""

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise RuntimeError("需要安装并配置 Gurobi。") from e

# =============================
# 0) 全局开关/超参数（你最常调的）
# =============================
NUM_SCENARIOS = 600
MC_PER_SCENARIO = 250

EPOCHS = 2500
BATCH_SIZE = 2048
LR = 1e-3

LAM_PHYS = 0.05

# Bayesian / VI
PRIOR_SIGMA = 1.0          # p(θ)=N(0, PRIOR_SIGMA^2)
BETA_KL_MAX = 1.0          # KL 权重的上限
KL_WARMUP_EPOCHS = 800     # 前 KL_WARMUP_EPOCHS 逐渐把 beta 从 0 -> BETA_KL_MAX
INIT_RHO = -5.0            # sigma=softplus(rho)，rho 越小初始 sigma 越小（更像点估计网络）
TRAIN_WEIGHT_SAMPLES = 1   # 每个 batch 用多少个 θ 采样估计 E_q[...]
EVAL_THETA_SAMPLES = 60    # 评估/画置信带时采样多少个 θ

PLOT_CI_BAND = True        # 是否画 95% 置信带
SHOW_MARK_005 = True       # 是否标记 y=0.05 分位点（论文风格）

SEED_DATA = 0
SEED_TRAIN = 0
SEED_EVAL = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


# =============================
# 1) 姜华3节点案例设置（按论文3.1.1）
# =============================
JIANG_CASE_MEAN = np.array([3.5, 1.9, 2.0], dtype=float)  # [mu_PD2, mu_PD3, mu_PR2]
USE_PLUS_005_FOR_PR2_STD = False

PF_LOAD = 0.9
PF_RE = 0.9
TAN_LOAD = math.tan(math.acos(PF_LOAD))
TAN_RE = math.tan(math.acos(PF_RE))

PG_MIN, PG_MAX = 1.0, 4.0
QG_MIN, QG_MAX = -2.5, 2.5

R12, X12 = 0.01, 0.03
R23, X23 = 0.01, 0.03

VMIN, VMAX = 0.95, 1.05
FMAX_P = 5.0
FMAX_Q = 5.0

PR2_MAX = 4.0


# =============================
# 2) Pomax 求解：max P0（姜华3节点）
# =============================
def solve_pomax_gurobi(pd2: float, pd3: float, pr2: float) -> float:
    qd2 = pd2 * TAN_LOAD
    qd3 = pd3 * TAN_LOAD
    qr2 = pr2 * TAN_RE

    m = gp.Model("jiang3bus_pomax")
    m.Params.OutputFlag = 0

    P0  = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P0")

    P12 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P12")
    Q12 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q12")
    P23 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P23")
    Q23 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q23")

    Pg  = m.addVar(lb=PG_MIN, ub=PG_MAX, name="Pg")
    Qg  = m.addVar(lb=QG_MIN, ub=QG_MAX, name="Qg")

    V2s = m.addVar(lb=VMIN**2, ub=VMAX**2, name="V2s")
    V3s = m.addVar(lb=VMIN**2, ub=VMAX**2, name="V3s")

    # PCC
    m.addConstr(P12 == -P0, name="PCC_P")
    m.addConstr(Q12 == 0.0, name="PCC_Q0")

    # KCL
    m.addConstr(P23 == P12 + pr2 - pd2, name="KCL_P_bus2")
    m.addConstr(Q23 == Q12 + qr2 - qd2, name="KCL_Q_bus2")

    m.addConstr(Pg == pd3 - P23, name="KCL_P_bus3")
    m.addConstr(Qg == qd3 - Q23, name="KCL_Q_bus3")

    # LinDistFlow: V^2
    m.addConstr(V2s == 1.0 - 2.0 * (R12 * P12 + X12 * Q12), name="Vdrop_12")
    m.addConstr(V3s == V2s - 2.0 * (R23 * P23 + X23 * Q23), name="Vdrop_23")

    # line limits
    m.addConstr(P12 <= FMAX_P); m.addConstr(P12 >= -FMAX_P)
    m.addConstr(P23 <= FMAX_P); m.addConstr(P23 >= -FMAX_P)
    m.addConstr(Q23 <= FMAX_Q); m.addConstr(Q23 >= -FMAX_Q)

    m.setObjective(P0, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return float("nan")
    return float(P0.X)


# =============================
# 3) 数据集：场景均值 + MC realization -> Pomax 样本集
# =============================
def sample_trunc_normal(mu: float, sigma: float, lo: float = 0.0, hi: float = None) -> float:
    x = np.random.normal(mu, sigma)
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return float(x)


def generate_dataset(num_scenarios=NUM_SCENARIOS, mc_per_scenario=MC_PER_SCENARIO, seed=SEED_DATA):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    X = np.zeros((num_scenarios, 3), dtype=float)
    Y = np.zeros((num_scenarios, mc_per_scenario), dtype=float)

    for i in range(num_scenarios):
        mu_PD2 = float(rng.uniform(2.0, 5.0))
        mu_PD3 = float(rng.uniform(1.0, 3.0))
        mu_PR2 = float(rng.uniform(0.5, 3.5))
        X[i, :] = [mu_PD2, mu_PD3, mu_PR2]

        std = 0.1 * np.array([mu_PD2, mu_PD3, mu_PR2], dtype=float)
        if USE_PLUS_005_FOR_PR2_STD:
            std[2] += 0.05

        for m in range(mc_per_scenario):
            y = float("nan")
            for _try in range(25):
                pd2 = sample_trunc_normal(mu_PD2, std[0], lo=0.0, hi=None)
                pd3 = sample_trunc_normal(mu_PD3, std[1], lo=0.0, hi=None)
                pr2 = sample_trunc_normal(mu_PR2, std[2], lo=0.0, hi=PR2_MAX)
                y = solve_pomax_gurobi(pd2, pd3, pr2)
                if np.isfinite(y):
                    break
            if not np.isfinite(y):
                y = solve_pomax_gurobi(mu_PD2, mu_PD3, min(mu_PR2, PR2_MAX))
            Y[i, m] = y

        if (i + 1) % 50 == 0:
            print(f"已生成 {i+1}/{num_scenarios} 组训练场景")

    return X, Y


# =============================
# 4) mean-only 物理损失（保持 v7 不变）
# =============================
def physics_loss_meanonly(x: torch.Tensor, mu_p0: torch.Tensor, alpha_phys: float = 1.0) -> torch.Tensor:
    mu_PD2, mu_PD3, mu_PR2 = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    mu_P12 = -mu_p0
    mu_P23 = mu_P12 + mu_PR2 - mu_PD2
    mu_Pg  = mu_PD3 - mu_P23

    mu_Q12 = mu_p0 * 0.0
    mu_Qr2 = mu_PR2 * TAN_RE
    mu_Qd2 = mu_PD2 * TAN_LOAD
    mu_Qd3 = mu_PD3 * TAN_LOAD
    mu_Q23 = mu_Q12 + mu_Qr2 - mu_Qd2
    mu_Qg  = mu_Qd3 - mu_Q23

    mu_V2s = 1.0 - 2.0 * (R12 * mu_P12 + X12 * mu_Q12)
    mu_V3s = mu_V2s - 2.0 * (R23 * mu_P23 + X23 * mu_Q23)

    relu = torch.relu
    loss = 0.0

    loss = loss + (relu(VMIN**2 - mu_V2s) ** 2).mean() + (relu(mu_V2s - VMAX**2) ** 2).mean()
    loss = loss + (relu(VMIN**2 - mu_V3s) ** 2).mean() + (relu(mu_V3s - VMAX**2) ** 2).mean()

    loss = loss + (relu(torch.abs(mu_P12) - FMAX_P) ** 2).mean() + (relu(torch.abs(mu_P23) - FMAX_P) ** 2).mean()
    loss = loss + (relu(torch.abs(mu_Q23) - FMAX_Q) ** 2).mean()

    loss = loss + (relu(PG_MIN - mu_Pg) ** 2).mean() + (relu(mu_Pg - PG_MAX) ** 2).mean()
    loss = loss + (relu(QG_MIN - mu_Qg) ** 2).mean() + (relu(mu_Qg - QG_MAX) ** 2).mean()

    return alpha_phys * loss


# =============================
# 5) GMM-2 likelihood / CDF（保持 v7 不变）
# =============================
def gmm2_log_prob(y: torch.Tensor, w: torch.Tensor, mu1: torch.Tensor, s1: torch.Tensor, mu2: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    z1 = (y - mu1) / s1
    z2 = (y - mu2) / s2
    logn1 = -0.5 * z1**2 - torch.log(s1) - 0.5 * math.log(2 * math.pi)
    logn2 = -0.5 * z2**2 - torch.log(s2) - 0.5 * math.log(2 * math.pi)

    logw = torch.log(w.clamp(min=1e-12))
    logp = torch.cat([logw[:, 0:1] + logn1, logw[:, 1:2] + logn2], dim=1)

    lp = torch.logsumexp(logp, dim=1, keepdim=True)
    lp = torch.clamp(lp, min=-1e6)
    return lp


def gmm2_cdf(z: np.ndarray, w: np.ndarray, mu1: float, s1: float, mu2: float, s2: float) -> np.ndarray:
    return w[0] * norm.cdf((z - mu1) / (s1 + 1e-12)) + w[1] * norm.cdf((z - mu2) / (s2 + 1e-12))


# =============================
# 6) VI：Bayesian Linear + KL(q||p)
# =============================
class BayesLinear(nn.Module):
    """
    权重/偏置的变分后验：q = N(mu, sigma^2)，sigma=softplus(rho)
    先验：p = N(0, prior_sigma^2)

    forward(sample=True): 采样权重（重参数技巧）
    forward(sample=False): 用 mu（后验均值）
    """
    def __init__(self, in_features, out_features, prior_sigma=1.0, init_rho=-5.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = float(prior_sigma)

        # Variational parameters
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_rho = nn.Parameter(torch.empty(out_features))

        # init
        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        nn.init.constant_(self.w_rho, init_rho)
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b_mu, -bound, bound)
        nn.init.constant_(self.b_rho, init_rho)

    def _sigma(self, rho):
        # sigma>0
        return torch.nn.functional.softplus(rho) + 1e-6

    def forward(self, x, sample: bool = True):
        if sample:
            eps_w = torch.randn_like(self.w_mu)
            eps_b = torch.randn_like(self.b_mu)
            w = self.w_mu + self._sigma(self.w_rho) * eps_w
            b = self.b_mu + self._sigma(self.b_rho) * eps_b
        else:
            w = self.w_mu
            b = self.b_mu
        return torch.nn.functional.linear(x, w, b)

    def kl_divergence(self):
        """
        KL( N(mu, sigma^2) || N(0, prior_sigma^2) ) closed-form.
        """
        prior_var = self.prior_sigma ** 2
        w_sigma = self._sigma(self.w_rho)
        b_sigma = self._sigma(self.b_rho)

        kl_w = (torch.log(torch.tensor(self.prior_sigma, device=self.w_mu.device)) - torch.log(w_sigma)).sum()
        kl_w = kl_w + 0.5 * ((w_sigma**2 + self.w_mu**2) / prior_var).sum()
        kl_w = kl_w - 0.5 * self.w_mu.numel()

        kl_b = (torch.log(torch.tensor(self.prior_sigma, device=self.b_mu.device)) - torch.log(b_sigma)).sum()
        kl_b = kl_b + 0.5 * ((b_sigma**2 + self.b_mu**2) / prior_var).sum()
        kl_b = kl_b - 0.5 * self.b_mu.numel()

        return kl_w + kl_b


class BayesGMM2Net(nn.Module):
    """
    v7 的 GMM2Net 结构不变：3->128->128->128->6
    只把 Linear 换成 BayesLinear，并提供 KL 总和。
    """
    def __init__(self, in_dim=3, hidden=128, depth=3, prior_sigma=1.0, init_rho=-5.0):
        super().__init__()
        self.depth = depth

        self.linears = nn.ModuleList()
        d = in_dim
        for _ in range(depth):
            self.linears.append(BayesLinear(d, hidden, prior_sigma=prior_sigma, init_rho=init_rho))
            d = hidden
        self.out = BayesLinear(d, 6, prior_sigma=prior_sigma, init_rho=init_rho)

        self.act = nn.ReLU()

    def forward(self, x, sample: bool = True):
        h = x
        for lin in self.linears:
            h = self.act(lin(h, sample=sample))
        out = self.out(h, sample=sample)

        logits = out[:, 0:2]
        w = torch.softmax(logits, dim=1)

        mu1 = out[:, 2:3]
        log_s1 = out[:, 3:4]
        mu2 = out[:, 4:5]
        log_s2 = out[:, 5:6]

        s1 = torch.nn.functional.softplus(log_s1) + 1e-3
        s2 = torch.nn.functional.softplus(log_s2) + 1e-3

        return w, mu1, s1, mu2, s2

    def kl_divergence(self):
        kl = 0.0
        for lin in self.linears:
            kl = kl + lin.kl_divergence()
        kl = kl + self.out.kl_divergence()
        return kl


# =============================
# 7) 训练：ELBO (NLL + physics + KL)
# =============================
def train_bayes_gmm2(X: np.ndarray, Y: np.ndarray,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                    lam_phys=LAM_PHYS,
                    prior_sigma=PRIOR_SIGMA, beta_kl_max=BETA_KL_MAX, kl_warmup_epochs=KL_WARMUP_EPOCHS,
                    seed=SEED_TRAIN):

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-9
    Xn = (X - X_mean) / X_std

    N, M = Y.shape
    X_flat = np.repeat(Xn, M, axis=0)
    y_flat = Y.reshape(-1, 1).astype(float)

    mask = np.isfinite(y_flat[:, 0])
    if not np.all(mask):
        dropped = int((~mask).sum())
        print(f"[warn] Dropping {dropped} non-finite Pomax samples before training.")
        y_flat = y_flat[mask]
        X_flat = X_flat[mask]

    X_t = torch.tensor(X_flat, device=DEVICE, dtype=torch.float32)
    y_t = torch.tensor(y_flat, device=DEVICE, dtype=torch.float32)

    X_mean_t = torch.tensor(X_mean, device=DEVICE, dtype=torch.float32)
    X_std_t  = torch.tensor(X_std,  device=DEVICE, dtype=torch.float32)

    net = BayesGMM2Net(prior_sigma=prior_sigma, init_rho=INIT_RHO).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    num_data = X_t.shape[0]  # 用于 KL 缩放：KL / N_data

    print("=== 开始训练 (B-PINN, GMM-2 NLL + KL-VI) ===")
    for ep in range(epochs):
        idx = rng.integers(0, num_data, size=batch_size)
        xb = X_t[idx]
        yb = y_t[idx]

        # KL warmup
        beta = beta_kl_max * min(1.0, (ep + 1) / max(1, kl_warmup_epochs))

        # 估计 E_q[ NLL + lam_phys*phys ]
        loss_nll_acc = 0.0
        loss_phys_acc = 0.0
        for _s in range(TRAIN_WEIGHT_SAMPLES):
            w, mu1, s1, mu2, s2 = net(xb, sample=True)
            logp = gmm2_log_prob(yb, w, mu1, s1, mu2, s2)
            loss_nll = (-logp).mean()

            mu_mix = w[:, 0:1] * mu1 + w[:, 1:2] * mu2
            x_raw = xb * X_std_t + X_mean_t
            loss_phys = physics_loss_meanonly(x_raw, mu_mix)

            loss_nll_acc = loss_nll_acc + loss_nll
            loss_phys_acc = loss_phys_acc + loss_phys

        loss_nll = loss_nll_acc / TRAIN_WEIGHT_SAMPLES
        loss_phys = loss_phys_acc / TRAIN_WEIGHT_SAMPLES

        # KL(q||p)
        kl = net.kl_divergence()
        loss_kl = beta * (kl / num_data)

        loss = loss_nll + lam_phys * loss_phys + loss_kl

        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at epoch {ep+1}, skip update.")
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if (ep + 1) % 200 == 0:
            print(f"Epoch {ep+1:4d} | loss={loss.item():.6f}  nll={loss_nll.item():.6f}  phys={loss_phys.item():.6f}  kl/N={ (kl/num_data).item():.6f}  beta={beta:.3f}")

    return net, X_mean, X_std


# =============================
# 8) 评估：后验预测 CDF（均值 + 置信带）+ ARMS + y=0.05
# =============================
def eval_and_plot_bayes(net, X_mean, X_std, mc_eval=6000, seed=SEED_EVAL,
                       theta_samples=EVAL_THETA_SAMPLES, plot_ci=PLOT_CI_BAND):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mu = JIANG_CASE_MEAN.copy()
    std = 0.1 * mu
    if USE_PLUS_005_FOR_PR2_STD:
        std[2] += 0.05

    # MC CDF
    y_list = []
    for _ in range(mc_eval):
        pd2 = sample_trunc_normal(mu[0], std[0], lo=0.0, hi=None)
        pd3 = sample_trunc_normal(mu[1], std[1], lo=0.0, hi=None)
        pr2 = sample_trunc_normal(mu[2], std[2], lo=0.0, hi=PR2_MAX)
        y = solve_pomax_gurobi(pd2, pd3, pr2)
        if np.isfinite(y):
            y_list.append(y)
    y_mc = np.array(y_list, dtype=float)
    y_mc.sort()

    z_min, z_max = float(y_mc.min()), float(y_mc.max())
    pad = 0.15 * (z_max - z_min + 1e-6)
    z_grid = np.linspace(z_min - pad, z_max + pad, 600)
    cdf_mc = np.searchsorted(y_mc, z_grid, side="right") / y_mc.size

    # 标准化输入（Jiang case）
    x = (mu.reshape(1, 3) - X_mean) / X_std
    xt = torch.tensor(x, device=DEVICE, dtype=torch.float32)

    # posterior predictive: sample theta -> cdf(z|theta)
    net.eval()
    cdf_samps = []
    q05_samps = []

    # 预先定 bracket（用于 root）
    # 用样本内范围 + 若干 sigma 做外延
    lo_base = z_min - 8.0 * pad
    hi_base = z_max + 8.0 * pad

    with torch.no_grad():
        for _ in range(theta_samples):
            w_t, mu1_t, s1_t, mu2_t, s2_t = net(xt, sample=True)
            w = w_t.cpu().numpy().reshape(-1)
            mu1 = float(mu1_t.cpu().numpy().reshape(-1)[0])
            mu2 = float(mu2_t.cpu().numpy().reshape(-1)[0])
            s1 = float(s1_t.cpu().numpy().reshape(-1)[0])
            s2 = float(s2_t.cpu().numpy().reshape(-1)[0])

            cdf_nn = gmm2_cdf(z_grid, w, mu1, s1, mu2, s2)
            cdf_samps.append(cdf_nn)

            if SHOW_MARK_005:
                y_mark = 0.05
                def f_root(z):
                    return float(gmm2_cdf(np.array([z]), w, mu1, s1, mu2, s2)[0] - y_mark)
                lo = lo_base - 4 * max(s1, s2)
                hi = hi_base + 4 * max(s1, s2)
                try:
                    q05 = float(brentq(f_root, lo, hi, maxiter=200))
                except Exception:
                    q05 = float("nan")
                q05_samps.append(q05)

    cdf_samps = np.array(cdf_samps, dtype=float)  # (S, len(z_grid))
    cdf_mean = np.nanmean(cdf_samps, axis=0)
    cdf_lo = np.nanpercentile(cdf_samps, 2.5, axis=0)
    cdf_hi = np.nanpercentile(cdf_samps, 97.5, axis=0)

    arms = 100.0 * math.sqrt(np.mean((cdf_mean - cdf_mc) ** 2))
    print(f"[B-PINN] Jiang case: ARMS(CDF) = {arms:.4f}%  (theta_samples={theta_samples})")

    # y=0.05 标记
    y_mark = 0.05
    q05_mc = float(np.quantile(y_mc, y_mark))

    if SHOW_MARK_005 and len(q05_samps) > 0:
        q05_samps = np.array(q05_samps, dtype=float)
        q05_samps = q05_samps[np.isfinite(q05_samps)]
        if q05_samps.size > 0:
            q05_mean = float(np.mean(q05_samps))
            q05_ci = (float(np.percentile(q05_samps, 2.5)), float(np.percentile(q05_samps, 97.5)))
        else:
            q05_mean = float("nan")
            q05_ci = (float("nan"), float("nan"))
        print(f"Quantile@0.05: MC={q05_mc:.4f},  BNN mean={q05_mean:.4f},  CI95%=[{q05_ci[0]:.4f},{q05_ci[1]:.4f}]")
    else:
        q05_mean = float("nan")
        q05_ci = (float("nan"), float("nan"))

    # plot (Fig.3-like: mean ± 2 std band + a few posterior sample curves)
    # 论文 Fig.3 用的是“predictive mean +/− two standard deviations”来画不确定性带
    # 这里对每个 z 上的 CDF 样本 {F^{(s)}(z)} 计算均值与标准差，再画 mean±2std
    cdf_std = np.nanstd(cdf_samps, axis=0)
    band_lo = np.clip(cdf_mean - 2.0 * cdf_std, 0.0, 1.0)
    band_hi = np.clip(cdf_mean + 2.0 * cdf_std, 0.0, 1.0)

    # Matplotlib 风格：尽量贴近论文排版
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "font.family": "serif",
    })

    fig = plt.figure(figsize=(10.5, 5.6), dpi=140)

    # 不确定性带（mean ± 2 std）
    plt.fill_between(z_grid, band_lo, band_hi, alpha=0.25, linewidth=0.0, label="B-PINN mean ± 2 std")

    # 叠加少量 posterior sample 曲线（淡灰），更接近 Fig.3 的“函数样本/不确定性”表达
    S = cdf_samps.shape[0]
    if S > 0:
        idx = np.linspace(0, S - 1, min(20, S), dtype=int)
        for k in idx:
            plt.plot(z_grid, cdf_samps[k], linewidth=0.8, alpha=0.12, color="0.4")

    # 参考 MC CDF（黑实线） + 预测均值（橙色虚线）
    plt.plot(z_grid, cdf_mc, linewidth=2.6, color="k", label="MC CDF")
    plt.plot(z_grid, cdf_mean, "--", linewidth=2.6, color="#d95f02",
             label=f"B-PINN mean (ARMS={arms:.3f}%)")

    # y=0.05 标记（与你之前一致）
    if SHOW_MARK_005:
        plt.axhline(y_mark, linestyle=":", linewidth=2.0, color="k", alpha=0.6)
        plt.scatter([q05_mc], [y_mark], s=70, marker="o", color="k", zorder=5)
        if np.isfinite(q05_mean):
            plt.scatter([q05_mean], [y_mark], s=80, marker="x", color="#d95f02", zorder=6)
            if np.isfinite(q05_ci[0]) and np.isfinite(q05_ci[1]):
                plt.hlines(y_mark, q05_ci[0], q05_ci[1], linewidth=2.2, color="#d95f02", alpha=0.9)

        plt.text(q05_mc, y_mark, f"  MC {q05_mc:.2f}", va="bottom", fontsize=12)
        if np.isfinite(q05_mean):
            plt.text(q05_mean, y_mark, f"  BNN {q05_mean:.2f}", va="top", fontsize=12)

    # 轴与标题
    plt.xlabel("P0 max (MW)")
    plt.ylabel("CDF")
    plt.ylim(-0.02, 1.02)
    plt.title("Pomax CDF (Jiang 3-bus settings) — B-PINN (VI) GMM2")

    # 论文风格通常不画网格
    plt.grid(False)
    plt.legend(loc="upper left", frameon=True)

    try:
        plt.tight_layout()
    except Exception as e:
        print(f"[plot warn] tight_layout failed: {e}")
    out_png = "Pomax_CDF_v8_Fig3Style_meanpm2std.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.show()



def main():
    print("正在生成训练数据集（按姜华设置 + Gurobi Pomax）...")
    X, Y = generate_dataset(num_scenarios=NUM_SCENARIOS, mc_per_scenario=MC_PER_SCENARIO, seed=SEED_DATA)

    net, X_mean, X_std = train_bayes_gmm2(
        X, Y,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        lam_phys=LAM_PHYS,
        prior_sigma=PRIOR_SIGMA, beta_kl_max=BETA_KL_MAX, kl_warmup_epochs=KL_WARMUP_EPOCHS,
        seed=SEED_TRAIN
    )

    eval_and_plot_bayes(net, X_mean, X_std, mc_eval=6000, seed=SEED_EVAL, theta_samples=EVAL_THETA_SAMPLES, plot_ci=PLOT_CI_BAND)


if __name__ == "__main__":
    main()
