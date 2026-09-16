#=
================================================================================
 LightDark1D + POMCPOW 测试脚本（带注释版）
================================================================================

 整体结构：
   1. 定义问题        pomdp   = LightDark1D()              [POMDPModels.jl]
   2. 设置求解器参数  solver  = POMCPOWSolver(...)         [POMCPOW.jl]
   3. 得到策略        planner = solve(solver, pomdp)       [POMCPOW.jl]
   4. 信念更新器      u_p     = BootstrapFilter(...)       [ParticleFilters.jl]
   5. 仿真器          hr      = HistoryRecorder(...)       [POMDPTools.jl]
   6. 仿真            history = simulate(hr, pomdp, planner, u_p)
   7. 计算回报        discounted_reward(history)          [POMDPTools.jl]
   8. 重复 nb_runs 次，取平均

 各包归属：
   - 属于 POMCPOW.jl 的：POMCPOWSolver、MaxUCB、solve 返回的 POMCPOWPlanner
   - BootstrapFilter 来自 ParticleFilters.jl（通用粒子滤波，不属于 POMCPOW）
   - HistoryRecorder / simulate / discounted_reward 来自 POMDPTools.jl
   - solve / simulate / action / update 是 POMDPs.jl 定义的统一接口函数

 在线算法说明：
   POMCPOW 是“在线”算法：solve() 并不真正求解，只是把参数和模型打包。
   真正的树搜索发生在 simulate() 内部，每一个真实步都会调用
   action(planner, b)，从当前信念 b 出发重新建树（本脚本每步约 3 秒）。

 每个真实步内部的流程（simulate 内部）：
   b  --action(planner, b)-->  a         # 树搜索，受 max_depth / max_time 控制
   s  --环境转移(hr.rng)---->  s', r, o  # 真实环境，由仿真器负责
   b  --update(u_p, b, a, o)-> b'        # 粒子滤波更新信念
   重复，直到终止状态或达到 max_steps

 三个“步数/深度”参数的区别：
   | 参数                     | 所属              | 控制什么                           |
   |--------------------------|-------------------|------------------------------------|
   | max_steps = 100          | HistoryRecorder   | 真实环境最多走多少步（外层循环）   |
   | max_depth = 30           | POMCPOWSolver     | 每次规划时模拟往前看多深（内层）   |
   | tree_queries / max_time  | POMCPOWSolver     | 每次规划做多少次模拟               |

 --------------------------------------------------------------------------------
 【记忆口诀】下棋类比
 --------------------------------------------------------------------------------
   horizon      : 这盘棋本来会下多久（问题本身的属性）
                  LightDark1D 是无限 horizon + 折扣 γ=0.9 + 有终止状态（动作 0）
   max_steps    : 裁判规定最多下 100 手（仿真时的人为截断，“有效 horizon”）
   max_depth    : 每走一手前，脑中最多往后推演几手（前瞻 horizon）
   tree_queries : 每走一手前，最多试想多少种变化
   max_time     : 每走一手的思考时间上限（棋钟）

   层级关系：
     horizon                       —— 问题本身有多长（理论）
     └─ max_steps                  —— 实际走几步（外层，真实环境）
        └─ 每一步之前思考：
           ├─ 次数/时间：tree_queries 或 max_time（先到先停）
           └─ 深度：     max_depth（每次推演最多几步）

   一句话：max_steps 管“走多少步”，max_depth 管“想多远”，
           tree_queries / max_time 管“想多少次、想多久”。

   截断是否合理，看折扣权重 γ^t：
     γ^30  = 0.9^30  ≈ 0.042    → max_depth = 30 之后的奖励权重已很小
     γ^100 = 0.9^100 ≈ 2.7e-5   → max_steps = 100 的截断误差可忽略
   （LightDark1D 通常十几步内就选动作 0 终止，很少走满 100 步）

 --------------------------------------------------------------------------------
 【max_time 的含义】
 --------------------------------------------------------------------------------
   max_time 是“每个真实决策步的思考预算”，只作用于 action(planner, b)：

     for t in 1:max_steps
         a = action(planner, b)     ← ★ 这里花 max_time = 3 秒建树
         s, o, r = 环境执行 a        ← 几乎不耗时
         b = update(u_p, b, a, o)   ← 粒子滤波，较快
     end

   1. 是“思考时间”而非“总时间”：
        单次 run 总耗时 ≈ 实际步数 × 3 秒 + 滤波时间
   2. anytime 特性：任何时刻停下都能给出动作；想得越久，树越大，决策通常越好
        → max_time 调节“决策质量 vs 计算成本”，可画“回报–max_time”曲线
   3. 模拟实时约束：例如机器人每步必须在限定时间内给出动作
   4. 公平比较的尺子：在线算法之间用相同的每步时间比较
   5. 代价是不可复现：按时间停止时，每步模拟次数取决于机器速度
        → 若需复现：max_time = Inf，并使用固定的 tree_queries

   让每步都跑满 3 秒：
     tree_queries = typemax(Int)   # 注意：不能写 Inf（Inf 是浮点数，类型不符）
     max_time     = 3.0
   - 本脚本的 10_000_000 在 3 秒内基本跑不完，效果上已等同于无穷
   - 超时检查在每次模拟之后进行，实际耗时会略超 3 秒
   - POMCPOW 计时可能使用 CPU 时间而非墙钟时间（请在源码 planner2.jl 中确认）
   - 跑满时间时树可能很大，大问题需注意内存

 --------------------------------------------------------------------------------
 【在线 vs 离线】两者代码框架相同（solve → simulate → 回报），区别在于计算落在哪一步
 --------------------------------------------------------------------------------
   |                 | 离线（SARSOP、MCVI、POMCGS） | 在线（POMCPOW、POMCP、DESPOT） |
   |-----------------|------------------------------|--------------------------------|
   | solve 做什么    | 真正求解，耗时长             | 只打包参数，几乎不耗时         |
   | 得到的 policy   | 完整策略（α 向量 / FSC）     | 没有预算好的策略               |
   | action(p, b)    | 查表 / 沿 FSC 走，很快       | 现场建树，耗时（本脚本 3 秒）  |
   | 计算覆盖范围    | 所有可能到达的信念           | 仅当前实际到达的信念           |
   | 能否复用        | 求解一次，多次仿真复用       | 每次仿真、每一步都要重新搜索   |
   比较时注意：离线的“总求解时间”应与在线的“累计每步搜索时间”对比。

 --------------------------------------------------------------------------------
 【与 POMCGS 的对比】（You et al., ICAPS 2025；POMCGraphSearch.jl）
 --------------------------------------------------------------------------------
   - 公开版本的 POMCGS 是离线算法，输出有限状态控制器（FSC）。
   - 论文中的“每步 3 秒”是给在线对比算法（AdaOPS / DESPOT / POMCPOW）的；
     POMCGS 在 LightDark 上使用的是 1 小时的总求解时间上限
     （对应仓库参数 max_planning_secs，默认 10000 秒）。
   - 论文中 LightDark 的结果：POMCGS 3.74，POMCPOW 3.29，AdaOPS 3.76。
   - 官方 README 未描述在线模式。论文仅设想：到达 FSC 叶节点时
     可改由在线求解器接管，以叶节点的信念作为初始信念。
   - 若所用版本有“在线模式”（每步 3 秒），推测其含义为：
       每个真实步从当前 FSC 节点出发运行 3 秒 UpdateFSC，
       再按 ψ(n) 取动作；图在各步之间复用并持续增长。
     与 POMCPOW 不同：POMCPOW 每步从头建树，上一步的计算不保留。
     → 需以实际源码中计时检查所在的循环为准。
   - 参数粗略对应：
       POMCGS num_fixed_observations (K, 观测聚类数)
         ≈ POMCPOW k_observation / alpha_observation（观测加宽上限）
       POMCGS max_search_depth ≈ POMCPOW max_depth
       POMCGS nb_particles / max_b_gap 在 POMCPOW 中无直接对应
================================================================================
=#

using POMCPOW                                 # POMCPOWSolver, MaxUCB
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
# POMDPModels: LightDark1D 模型
# POMDPTools : HistoryRecorder, discounted_reward 等仿真工具
# POMDPs     : solve / simulate / action / update 等接口
using CSV, DataFrames
using Random
using Dates
using ParticleFilters                         # BootstrapFilter

Random.seed!(1)

# ---------------------------------------------------------------------------
# 1. 问题定义
#    LightDark1D：一维位置，靠近“亮区”时观测噪声小；选择动作 0 时终止，
#    若此时位置接近目标则得正奖励（+10），否则得负奖励。
#    动作空间为离散的 {-1, 0, 1}，观测为连续值，折扣因子 discount = 0.9。
# ---------------------------------------------------------------------------
pomdp = LightDark1D()

mx_depth    = 30      # 树搜索的最大深度（内层，每次规划往前看多少步）
mx_steps    = 100     # 真实环境中最多执行多少步（外层，单次仿真的步数上限）
nb_runs     = 10      # 独立仿真次数
n_particles = 5000    # 粒子滤波的粒子数

# ---------------------------------------------------------------------------
# 2. 求解器参数（属于 POMCPOW.jl）
#    POMCPOWSolver 只是参数容器，本身不做计算。
# ---------------------------------------------------------------------------
solver = POMCPOWSolver(
    # 树内选动作的准则：UCB1
    #   Q(b,a) + c * sqrt(log N(b) / N(b,a))，这里 c = 90
    #   LightDark 奖励量级约 ±10，故探索常数取较大值。默认 MaxUCB(1.0)
    criterion       = MaxUCB(90.0),

    # 观测的渐进加宽（progressive widening）：
    #   当 (b,a) 节点访问次数为 N 时，只有观测子节点数 <= k * N^alpha
    #   才允许新增观测分支，否则从已有分支中重复采样。
    #   alpha = 1/15 时，即使 N = 10^6，N^(1/15) ≈ 2.5，
    #   上限约 12~13 个观测分支 → 树“窄而深”。
    #   粗略类比 POMCGS 的 num_fixed_observations = 10。
    #   默认 k_observation = 10, alpha_observation = 0.5
    k_observation   = 5.0,
    alpha_observation = 1/15,

    # 搜索树最大深度（包括树内选择 + 叶节点之后的 rollout）
    # 注意：它与 HistoryRecorder 的 max_steps 不是一回事
    max_depth = mx_depth,

    # 两个停止条件，先达到哪个就停：
    #   tree_queries：每次规划的模拟次数上限（这里设很大）
    #   max_time    ：每次规划的时间上限（秒）→ 实际由它决定
    # 注意：以时间为停止条件时，结果受机器负载影响，无法完全复现。
    #       若需可复现，可设 max_time = Inf 并使用固定的 tree_queries。
    # 若想明确表达“每步跑满 3 秒”，可改为：
    #   tree_queries = typemax(Int),   # 不能写 Inf（Inf 是 Float64，类型不符）
    tree_queries = 10_000_000,  # 设得足够大，让 max_time 成为实际的停止条件
    max_time = 3.0,

    # 不检查新采样的观测是否与已有分支相同。
    # LightDark1D 观测为连续值，几乎不会重复，关掉可省去哈希开销。
    check_repeat_obs  = false,

    # 搜索内部使用的随机数生成器。
    # 注意：它在所有 run 之间共享，第 i 次运行会受前面运行的影响。
    rng = MersenneTwister(1)

    # 未显式设置但起作用的默认参数：
    #   estimate_value   = 随机策略 rollout（叶节点价值估计）
    #   enable_action_pw = true, k_action = 10, alpha_action = 0.5
    #                      （动作渐进加宽；本问题只有 3 个动作，影响不大）
    #   final_criterion  = MaxQ()（搜索结束后选 Q 值最大的动作执行）
)

# ---------------------------------------------------------------------------
# 3. 得到策略（属于 POMCPOW.jl）
#    返回 POMCPOWPlanner，它是一个 Policy。
#    在线算法：这一步不做搜索，只是打包 solver 与 pomdp。
#    真正的搜索在 simulate 中每一步调用 action(planner, b) 时发生，
#    b 必须是可以 rand() 采样的分布（这里是粒子集合）。
# ---------------------------------------------------------------------------
planner = solve(solver, pomdp)


results = Float64[]

for i in 1:nb_runs
    # -----------------------------------------------------------------------
    # 4. 信念更新器（ParticleFilters.jl，不属于 POMCPOW）
    #    Bootstrap / SIR 粒子滤波。输入：
    #      pomdp       ：用其转移函数推进粒子，用观测概率 pdf 给粒子加权
    #      n_particles ：粒子数
    #      rng         ：滤波器内部随机数（转移采样与重采样）
    #    仿真开始时从 initialstate(pomdp) 采样 n_particles 个粒子作为初始信念；
    #    之后每步 update(u_p, b, a, o)：推进 → 按观测似然加权 → 低方差重采样
    # -----------------------------------------------------------------------
    u_p = BootstrapFilter(pomdp, n_particles, MersenneTwister(1000 + i))

    # -----------------------------------------------------------------------
    # 5. 仿真器（POMDPTools.jl）
    #    扮演“真实环境”，并记录完整轨迹。输入：
    #      max_steps：真实环境最多执行多少步（外层循环）
    #                 LightDark1D 选动作 0 即终止，通常会提前结束
    #      rng      ：真实初始状态、真实转移、真实观测的随机性
    #    若只需要回报、不需要轨迹，可改用更快的 RolloutSimulator：
    #      ro = RolloutSimulator(max_steps = mx_steps, rng = ...)
    #      run_return = simulate(ro, pomdp, planner, u_p)
    # -----------------------------------------------------------------------
    hr = HistoryRecorder(max_steps = mx_steps, rng = MersenneTwister(2000 + i))

    # -----------------------------------------------------------------------
    # 6. 仿真
    #    输入：仿真器、模型、策略、信念更新器
    #    （可选再传初始信念 b0、初始状态 s0）
    #    hr 是“仿真器”，history 是它跑出的“结果”（SimHistory），
    #    history 中含每步的 s, a, r, o, b 等信息，可这样查看：
    #      for step in eachstep(history, "s,a,r,o")
    #          println(step)
    #      end
    # -----------------------------------------------------------------------
    history = simulate(hr, pomdp, planner, u_p)

    # 7. 折扣回报：sum_t discount(pomdp)^t * r_t（LightDark1D 中 discount = 0.9）
    run_return = discounted_reward(history)
    push!(results, run_return)
    println("Run $i: $run_return")
end

# ---------------------------------------------------------------------------
# 8. 统计与保存结果
# ---------------------------------------------------------------------------
df = DataFrame(run_id = 1:nb_runs, return_value = results)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
#CSV.write("data_POMCPOW/LD_pomcpow_$(timestamp)_$(mean(results)).csv", df)
# 注意：上面的 CSV.write 被注释掉了，下面最后一行的“Results saved”提示并不准确

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results))")
println("Results saved to LD_pomcpow_$(timestamp)_$(mean(results)).csv")