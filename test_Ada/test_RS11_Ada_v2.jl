using AdaOPS
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using RockSample
using CSV, DataFrames
using Random
using Dates
using ParticleFilters
using Statistics

# 本文件由 test_RS15_Ada_v2.jl 派生: 除了 RockSamplePOMDP(11, 11) 之外,
# 所有实验环境参数和超参数与 RS15 版本逐字相同, 这样 RS11/RS15 的差异
# 只来自问题规模本身。改 RS15 的参数时记得同步改这里。

# ============================================================================
# 实验环境一致性基准 (baseline = test_POMCPOW/test_RS15_POMCPOW.jl)
#   mx_depth    = 40    -> POMCPOW: max_depth / AdaOPS: max_depth / ARDESPOT: D
#   mx_steps    = 100   -> HistoryRecorder(max_steps = ...)
#   nb_runs     = 100   -> 正式实验值 (当前文件若为 1/2 是 smoke test, 出数据前必须改回 100)
#   n_particles = 5000  -> 仿真中跟踪真实 belief 的 BootstrapFilter 粒子数
#   规划预算     = 3.0s  -> POMCPOW: max_time / AdaOPS: T_max / ARDESPOT: T_max
#   种子         = planner MersenneTwister(1), filter 1000+i, recorder 2000+i
#   LaserTag 环境实例 = gen_lasertag(rng = MersenneTwister(7), robot_position_known=false)
# 以上七项在 test_POMCPOW / test_Ada / test_ARDESPOT 三个目录里必须完全一致。
# ============================================================================

# 本文件按 test_POMCPOW/test_RS15_POMCPOW.jl 的程序结构重写 (BootstrapFilter + HistoryRecorder
# + discounted_reward + 两个 CSV), 只把 solver 换成 AdaOPS, 方便与 POMCPOW 结果一一对比。
# 原 test_RS11_Ada.jl 保持不动。

Random.seed!(1)

#POMDP problem
pomdp = RockSamplePOMDP(11, 11)

mx_depth = 40 #defult 10, try fixed  —— 这里对应 AdaOPS 的 max_depth (搜索树最大深度, package default 90)
mx_steps = 100
nb_runs   = 100
n_particles = 5000

# --- bounds: AdaOPS 必须提供上下界, POMCPOW 不需要 ---
# 取自 JuliaPOMDP/AdaOPS.jl README 的 RockSample 例子。
bounds = AdaOPS.IndependentBounds(
    FORollout(RSExitSolver()),   # lower bound: 直接走向出口的启发式 rollout
    FOValue(RSMDPSolver()),      # upper bound: 完全可观测 MDP 的最优值
    check_terminal = true,
    consistency_fix_thresh = 1e-5,
)

# ---- 上下界变体 B [已注释]: 无信息上下界, 用来做"下界质量影响多大"的消融 ----
# 启用变体 B 时, 必须同时在 test_ARDESPOT/test_RS11_ARDESPOT.jl 里切到同名变体 B,
# 否则 AdaOPS 和 AR-DESPOT 拿到的先验信息量不同, 两边的结果不可比。
# bounds = AdaOPS.IndependentBounds(
#     FORollout(RandomPolicy(pomdp, rng = MersenneTwister(3))),  # 随机 rollout 下界
#     10.0 / (1.0 - discount(pomdp)),                            # 常数上界 Vmax = Rmax/(1-gamma)
#     check_terminal = true,
#     consistency_fix_thresh = 1e-5,
# )

#POMDP solver: just for hyperparameter tuning, not for actual planning
# 超参数取自 AdaOPS.jl README 的 RockSample 例子 (delta/m_min/m_max/zeta/num_b)。
solver = AdaOPSSolver(
    bounds = bounds,
    delta  = 0.3,            # README default for RockSample: δ-packing 阈值
    m_min  = 30,             # 自适应粒子数下限 (KLD-sampling)
    m_max  = 200,            # 自适应粒子数上限
    zeta   = 0.1,            # KLD-sampling 的目标散度
    num_b  = 10_000,         # backup 预分配的 belief 数
    max_depth      = mx_depth,       # 搜索树最大深度, 与 POMCPOW 的 max_depth 对齐
    T_max  = 3.0,            # 每步在线规划时间预算, 与 POMCPOW 的 max_time = 3.0 一致
    tree_in_info = false,
    rng = MersenneTwister(1),
)

#policy,
planner = solve(solver, pomdp)

results = Float64[]

for i in 1:nb_runs
    u_p = BootstrapFilter(pomdp, n_particles, MersenneTwister(1000 + i)) #belief update
    hr = HistoryRecorder(max_steps = mx_steps, rng = MersenneTwister(2000 + i)) #simulates policy execution in "real" environment, record  trajectory
    history = simulate(hr, pomdp, planner, u_p) #

    run_return = discounted_reward(history)
    push!(results, run_return)
    println("Run $i: $run_return")
end

# Save results to CSV
df = DataFrame(run_id = 1:nb_runs, return_value = results)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
CSV.write("data_Ada/RS11_delta(0.3)_adaops_$(timestamp)_$(mean(results))_R1.csv", df)

# 均值和标准差单独存一个 CSV
summary_df = DataFrame(mean_return = mean(results), std_return = std(results))
CSV.write("data_Ada/RS11_delta(0.3)_adaops_$(timestamp)_summary_R1.csv", summary_df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results)), Std: $(std(results))")




#changes 1st round:
#delta=0.3, m_min=30, m_max=200, zeta=0.1, num_b=10000 (AdaOPS.jl README RockSample 设置)
#D=40, T_max=3.0s, 100 runs, 5000 tracking particles, max_steps=100
