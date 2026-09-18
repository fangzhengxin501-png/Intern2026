# 实验记录 — POMCPOW / AdaOPS / AR-DESPOT

> 超参数以各 `test_*/test_*.jl` 脚本里当前启用（未注释）的值为准，最后核对日期：2026-09-17。
> 详细的轮次日志、结果主表见 `docs/experiment_log.md`；本文件只记「环境 + 超参数」。

---

## 0. 固定实验环境（三个算法、四个问题全部一致，不随实验改变）

基准脚本：`test_POMCPOW/test_RS15_POMCPOW.jl`。以下任意一项改动都要在 `docs/experiment_log.md` 里开新一轮。

| 项目 | 值 | 各包对应字段 |
|---|---|---|
| 搜索深度 | 40 | POMCPOW `max_depth` / AdaOPS `max_depth` / ARDESPOT `D` |
| 每步规划预算 | 3.0 s | POMCPOW `max_time` / AdaOPS `T_max` / ARDESPOT `T_max` |
| 单局最大步数 | 100 | `HistoryRecorder(max_steps = 100)` |
| 重复次数 | 100 | `nb_runs`（12 个脚本当前均为 100；smoke test 时临时设 1，之后必须改回） |
| 跟踪粒子数 | 5000 | `BootstrapFilter(pomdp, 5000, ...)` |
| 迭代次数上限 | 不设限 | POMCPOW `tree_queries` / ARDESPOT `max_trials` = `typemax(Int)`，由时间预算停止 |
| 随机种子 | planner `MersenneTwister(1)`；belief filter `1000 + i`；HistoryRecorder `2000 + i`；全局 `Random.seed!(1)` | 第 i 局 |
| 回报口径 | `discounted_reward(history)`（折扣回报） | — |
| 观测分箱（仅 AR-DESPOT × LightDark） | `OBS_BIN = 1.0` | `BinnedObsPOMDP` wrapper，只作用于 planner 模型 |

### 四个问题的固定属性（均为包默认值，脚本未覆盖）

| 问题 | 构造 | γ | 关键回报 | 观测空间 |
|---|---|---|---|---|
| RS11 | `RockSamplePOMDP(11, 11)` | 0.95 | good rock +10 / bad rock −10 / exit +10 | 离散 |
| RS15 | `RockSamplePOMDP(15, 15)` | 0.95 | 同上 | 离散 |
| LaserTag | `gen_lasertag(rng = MersenneTwister(7), robot_position_known = false)` | 0.95 | tag +10 / 误 tag −10 / 每步 −1 | 离散（DESPOTEmu / DMeas） |
| LightDark | `LightDark1D()` | 0.9 | correct +10 / incorrect −10 | **连续**（AR-DESPOT 经 planner 侧分箱后适用） |

> γ 跨问题不同是 benchmark 自身属性，**绝对回报不可跨问题比较**，只在同一问题内比三个算法。
> 深度 40 对 γ = 0.95 截断约 13% 的价值（0.95⁴⁰ ≈ 0.129），三个算法同等截断。

### 运行环境 / 文件对照

| 问题 | POMCPOW | AdaOPS | AR-DESPOT |
|---|---|---|---|
| RS11 | `test_POMCPOW/test_RS11_POMCPOW.jl` | `test_Ada/test_RS11_Ada_v2.jl` | `test_ARDESPOT/test_RS11_ARDESPOT.jl` |
| RS15 | `test_POMCPOW/test_RS15_POMCPOW.jl` | `test_Ada/test_RS15_Ada_v2.jl` | `test_ARDESPOT/test_RS15_ARDESPOT.jl` |
| LaserTag | `test_POMCPOW/test_LaserTag_POMCPOW.jl` | `test_Ada/test_LaserTag_Ada.jl` | `test_ARDESPOT/test_LaserTag_ARDESPOT.jl` |
| LightDark | `test_POMCPOW/test_LightDark_POMCPOW.jl` | `test_Ada/test_LightDark_Ada_v2.jl` | `test_ARDESPOT/test_LightDark_ARDESPOT.jl` |

Julia 环境：`Environments_library/Env_POMCPOW` · `Env_Ada` · `Env_ARDESPOT`（LaserTag 需先 `] add https://github.com/JuliaPOMDP/LaserTag.jl.git`）。
输出目录：`data_POMCPOW/` · `data_Ada/` · `data_ARDESPOT/`，**必须在各自目录下启动 Julia**（脚本用相对路径）。

---

## 0.5 超参数分层：哪些固定、哪些按问题调

后面三张表要配合这个分层读。**只有第三层是真正的调参旋钮**，前两层跨实验一律固定。

### 第一层：协议参数 —— 全部实验、全部算法锁死

即上面第 0 节的七项，加上 `tree_in_info = false`、bounds 的 `check_terminal = true` 与
`consistency_fix_thresh = 1e-5`。这层不是超参数，是「给多少资源、怎么量结果」。
改任意一项都要在 `docs/experiment_log.md` 里开新的一轮。

### 第二层：由问题性质决定 —— 按明文规则固定，不许当旋钮调

这层参数跨问题确实取不同值，但那**不是调参，是算法要正确实例化的前提**。
它们由规则决定，不该试出来：

| 算法 | 参数 | 规则 |
|---|---|---|
| POMCPOW | `check_repeat_obs` | 离散观测 → `true`；连续观测 → `false` |
| POMCPOW | `check_repeat_act` | 离散动作 → `true` |
| POMCPOW | `enable_action_pw` | 动作空间小且离散 → `false`（不对动作做渐进加宽） |
| AdaOPS | `grid` | 仅连续状态需要（LD 用 `StateGrid`）；离散状态不设 |
| AdaOPS | `Base.convert` 方法 | 连续状态需要 `convert(::Type{SVector{N,Float64}}, s)` 才能做 KLD-sampling 计数 |
| AdaOPS | `m_min` / `m_max` | 按该问题 belief 的宽度定（LaserTag 的对手位置比 RockSample 分散 → 50/500），**扫参时不动** |
| ARDESPOT | `bounds_warnings` | 默认 `true`；仅在已知界不一致会刷屏时关掉 |
| ARDESPOT | `OBS_BIN`（LD 专有） | 连续观测必须分箱才能用 DESPOT；宽度本身属第三层 |
| 三者 | bounds 变体 A / B | **同一问题的三个文件必须同步切** |

**上下界属于这一层，不属于第三层。** 它们不是用来刷分的旋钮，而是保证三个算法拿到同等
先验信息的契约；换个更强的下界去提高某一列的分数，等于给那个算法开小灶。

> 各包的包装类型名不同但语义相同，不要误以为口径不一致：
> `FORollout`（AdaOPS / BasicPOMCP）≡ `DefaultPolicyLB`（ARDESPOT）——同一策略的 rollout 下界；
> `FOValue`（AdaOPS）≡ `FullyObservableValueUB`（ARDESPOT）——同一个完全可观测 MDP 值上界。
> 唯一无法消除的差异是实现层面：DESPOT 在**确定化的 K 条 scenario** 上算界，
> AdaOPS 在**带权自适应粒子集**上算界，估计量的方差结构不同，属算法结构性差异。

### 第三层：真正的调参旋钮 —— 按问题调，但投入必须对等

**每个算法只认一个主旋钮**，每个问题扫同样数量的值（4–5 个点），用与正式实验**分开**的
调参种子（建议 filter `5000+i`、recorder `6000+i`），扫参结果全部留档。
这样「三个算法调参投入对等」才是一句可核实的话。

| 算法 | 主旋钮（每问题扫同样数量的值） | 次要旋钮（锁在包默认值，声明不调） |
|---|---|---|
| POMCPOW | `criterion` 的 UCB 常数 c | `k_observation` = 10.0 · `alpha_observation`（离散 0.5，连续另定） |
| AdaOPS | `delta`（约 [0.05, 0.3]） | `zeta` = 0.1 · `num_b` = 10000 |
| AR-DESPOT | `K` | `lambda` = 0.01 · `xi` = 0.95 · `epsilon_0` = 0.0 |

**当前调参投入并不对等**，出正式对比前需要拉平：

| 算法 | 主旋钮现状 |
|---|---|
| POMCPOW | 已从默认 1.0 改到 20.0，但**未扫参**；且 RS 上疑似过大（见下方零回报问题） |
| AdaOPS | RS 用 README 原值 0.3，LaserTag / LD 调到 0.1，**未系统扫参** |
| AR-DESPOT | 四个问题全为默认 `K = 500`，**完全未调** |

---

## POMCPOW

固定项：`max_depth = 40`、`max_time = 3.0`、`tree_queries = typemax(Int)`、`rng = MersenneTwister(1)`。
POMCPOW 无上下界，`estimate_value` 承担与 AdaOPS/ARDESPOT **下界**相同的角色，必须与它们对齐。

| 实验 | criterion | k_observation | alpha_observation | check_repeat_obs | 其他 | estimate_value（= 下界） |
|---|---|---|---|---|---|---|
| RS11 | `MaxUCB(1.0)` | 10.0 | 0.5 | true | — | `FORollout(RSExitSolver())` |
| RS15 | `MaxUCB(1.0)` | 10.0 | 0.5 | true | — | `FORollout(RSExitSolver())` |
| LaserTag | `MaxUCB(1.0)` | 10.0 | 0.5 | true | `check_repeat_act = true`、`enable_action_pw = false` | `FORollout(RandomPolicy(pomdp, rng = MersenneTwister(3)))` |
| LightDark | `MaxUCB(1.0)` | 10.0 | **1/15** | **false** | — | `FORollout(FunctionPolicy(s -> s.y < 0 ? 1 : -1))` |

备注：
- R1（2026-09-17）：四个问题的 UCB 常数统一回到 POMCPOW.jl README 默认值 **1.0**（原为 20.0）。
- LightDark 连续观测：`alpha_observation = 1/15`、关闭 `check_repeat_obs`。
- 2026-09-17：LaserTag 原先未设 `estimate_value`，靠 POMCPOW 的默认随机 rollout。对随机策略而言它与 Ada/ARDESPOT 的 `RandomPolicy` 下界在分布上等价，但默认值的 rng 由包内部构造（不是统一的种子 3），且依赖包默认值不稳定，故改为显式写出。
- 待办：LaserTag 的 `MaxUCB(20.0)` 相对其回报量级（−20..+10）偏大，计划扫 [3, 5, 10, 20]。

---

## AdaOPS

固定项：`max_depth = 40`、`T_max = 3.0`、`zeta = 0.1`、`num_b = 10_000`、`tree_in_info = false`、`rng = MersenneTwister(1)`；
bounds 统一用 `AdaOPS.IndependentBounds(..., check_terminal = true, consistency_fix_thresh = 1e-5)`。

| 实验 | delta | m_min | m_max | grid | 下界 | 上界 |
|---|---|---|---|---|---|---|
| RS11 | 0.3 | 30 | 200 | — | `FORollout(RSExitSolver())` | `FOValue(RSMDPSolver())` |
| RS15 | 0.3 | 30 | 200 | — | `FORollout(RSExitSolver())` | `FOValue(RSMDPSolver())` |
| LaserTag | 0.3 | 50 | 500 | — | `FORollout(RandomPolicy(pomdp, rng = MersenneTwister(3)))` | 常数 `10.0`（= tag_reward） |
| LightDark | 0.3 | 100 | 5000 | `StateGrid(collect(-20.0:1.0:20.0))` | `FORollout(FunctionPolicy(s -> s.y < 0 ? 1 : -1))` | 常数 `10.0`（= correct_r） |

备注：
- R1（2026-09-17）：四个问题的 `delta` 统一为 **0.3**（AdaOPS.jl README RockSample 例子的值；LaserTag/LD 原为 0.1）。`delta` 是主旋钮，范围约 [0.05, 0.3]，R2 扫参。
- LaserTag 的 belief（对手位置）比 RockSample 分散，所以 m_min/m_max 放大到 50/500。
- LightDark 需要 `Base.convert(::Type{SVector{1,Float64}}, s::LightDark1DState)` 才能做 KLD-sampling 计数。
- 消融变体 B（已在脚本中注释）：RS11/RS15 换成 `FORollout(RandomPolicy(rng = MersenneTwister(3)))` + 常数上界 `10/(1-γ) = 200.0`；LightDark 换成常数 −10.0 / 10.0。切变体 B 时必须同步切对应的 ARDESPOT 文件。

---

## ARDESPOT

固定项：`D = 40`、`T_max = 3.0`、`K = 500`、`lambda = 0.01`、`xi = 0.95`、`epsilon_0 = 0.0`、
`max_trials = typemax(Int)`、`tree_in_info = false`、`rng = MersenneTwister(1)`（均为 ARDESPOT.jl 默认值，只改 T_max）；
bounds 统一用 `IndependentBounds(..., check_terminal = true, consistency_fix_thresh = 1e-5)`。

| 实验 | K | 下界 | 上界 | bounds_warnings |
|---|---|---|---|---|
| RS11 | 500 | `DefaultPolicyLB(RSExitSolver())` | `FullyObservableValueUB(RSMDPSolver())` | true |
| RS15 | 500 | `DefaultPolicyLB(RSExitSolver())` | `FullyObservableValueUB(RSMDPSolver())` | true |
| LaserTag | 500 | `DefaultPolicyLB(RandomPolicy(pomdp, rng = MersenneTwister(3)))` | 常数 `10.0` | true |
| LightDark | 500 | `DefaultPolicyLB(FunctionPolicy(s -> s.y < 0 ? 1 : -1))` | 常数 `10.0` | true |

备注：
- `K`（场景数）是主要调参项，常用 [100, 500]。
- **LightDark 观测分箱（2026-09-17 改）**：DESPOT 按观测精确相等分组，连续观测会让每个观测分支只剩 1 条 scenario，树退化为 K 条独立单粒子轨迹。现用 `BinnedObsPOMDP` wrapper 把观测离散成 bin 索引，**只交给 planner**；`HistoryRecorder` 的仿真环境与 `BootstrapFilter` 仍用原始连续 `LightDark1D()`。问题本身没变，三列仍可比；离散化算作算法的一部分，报告时须写明这是「AR-DESPOT + 观测分箱」变体。
  - `OBS_BIN = 1.0`（依据：判定成功容差 |y| < 1，分辨率与决策精度同量级），属第三层调参项，扫 [0.5, 1.0, 2.0]；设 `OBS_BIN = nothing` 可退回退化基线做对照。
  - 观测噪声随 y 变化（离光源越远越糊），固定宽度必然有偏；文件内注释了按噪声尺度缩放的**相对分箱**变体。
  - 同时修正了该文件把 `DESPOTSolver` 的 `D` 误写成 `max_depth` 的问题（`@with_kw` 会判为未知关键字直接报错），并把 `bounds_warnings` 恢复为 `true`。
- 上界口径只要求与 AdaOPS 严格一致（POMCPOW 无上界，属结构性差异）。

---

## R1（第 1 轮，初始版）配置口径

三个主旋钮一律取**包默认 / README 值**，跨四个问题统一，不做任何按问题的调参 —— 即 §0.5 里说的
「主表：默认配置（零调参）」。CSV 文件名统一带 `_R1` 后缀。

| 算法 | 主旋钮 | R1 取值 | 来源 |
|---|---|---|---|
| POMCPOW | `criterion` | `MaxUCB(1.0)` | POMCPOW.jl README 默认值 |
| AdaOPS | `delta` | 0.3 | AdaOPS.jl README（RockSample 例子） |
| AR-DESPOT | `K` | 500 | ARDESPOT.jl 默认值；DESPOT 论文 §5.1「we chose K = 500」 |

次要旋钮（锁定，声明不调）：`lambda = 0.01`、`xi = 0.95`、`epsilon_0 = 0.0`（ARDESPOT.jl 默认值）。
注：DESPOT 论文 JAIR 版 Table 2 在 RockSample 上选的是 `lambda = 0.0`、LaserTag 为 `0.01`；R1 未采用，留待 R2。

| 状态 | 问题 × 算法 | 情况 |
|---|---|---|
| ✅ 已出数 | RS11 / RS15 / LaserTag / LightDark × POMCPOW、AdaOPS；RS11 / RS15 × AR-DESPOT | — |
| ❌ 未出数 | LaserTag × AR-DESPOT | 跑到中途报错 |
| ❌ 未出数 | LightDark × AR-DESPOT | 观测分箱尚未正常工作 |

第二层（由问题性质决定）的参数仍按各自规则取值，未统一：
POMCPOW 的 `alpha_observation`（LD = 1/15）、`check_repeat_obs`；
AdaOPS 的 `m_min`/`m_max`（RS 30/200、LaserTag 50/500、LD 100/5000）与 `grid`；
ARDESPOT 的 `OBS_BIN`（LD = 1.0）。

## 待办

- [ ] `test_LaserTag_ARDESPOT.jl`：R1 跑到中途报错，抓 stack trace 后修复并补 R1 数据
- [ ] `test_LightDark_ARDESPOT.jl`：R1 的观测分箱（`BinnedObsPOMDP`，`OBS_BIN = 1.0`）未正常工作，调通后补 R1 数据
- [ ] R2 按 §0.5 拉平调参投入：POMCPOW 扫 `MaxUCB` [1, 3, 5, 10]、AdaOPS 扫 `delta` [0.05, 0.1, 0.2, 0.3]、AR-DESPOT 扫 `K` [125, 250, 500, 1000]，各问题同等点数、用调参专用种子（filter `5000+i` / recorder `6000+i`）
- [ ] R2 顺带核 `lambda`：论文 RS = 0.0、LaserTag = 0.01，与当前统一的 0.01 不符
- [ ] LightDark 分箱宽度扫 [0.5, 1.0, 2.0]，并与 `OBS_BIN = nothing` 的退化基线对照
- [ ] 三套脚本加单局计时，确认 3 s 预算是否被用满
- [ ] `Env_ARDESPOT` 仍需 `] add https://github.com/JuliaPOMDP/LaserTag.jl.git`
- [ ] `test_RoombaL_POMCPOW.jl` 缺 `RoombaPOMDPs` 依赖，`LidarRoombaPOMDP()` 未定义，暂时搁置
