using AdaOPS
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using StaticArrays
using CSV, DataFrames
using Random
using Dates
using ParticleFilters
using Statistics

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

# 本文件按 test_POMCPOW/test_LightDark_POMCPOW.jl 的程序结构重写 (BootstrapFilter +
# HistoryRecorder + discounted_reward + 两个 CSV), solver 换成 AdaOPS。
# 原 test_LightDark_Ada.jl 保持不动。

Random.seed!(1)

#POMDP problem
pomdp = LightDark1D()

# AdaOPS 的自适应粒子滤波 (KLD-sampling) 需要把状态映射到向量, 才能落到 StateGrid 的格子里计数。
# LightDark1D 的状态只有一个连续维度 y。
Base.convert(::Type{SVector{1, Float64}}, s::LightDark1DState) = SVector(convert(Float64, s.y))

mx_depth = 40 #defult 10, try fixed 40 —— 对应 AdaOPS 的 max_depth (搜索树最大深度, package default 90)
mx_steps = 100
nb_runs   = 100
n_particles = 5000

# --- bounds: AdaOPS 必须提供上下界, POMCPOW 不需要 ---
# lower: 一个简单启发式 rollout, 始终朝光源 (y = 0) 方向走。
# upper: LightDark1D 的单步最大回报 correct_r = 10.0 (POMDPModels 默认值), 作为常数上界。
# 上下界越紧 AdaOPS 效果越好, 之后可以换成更紧的 FOValue。
lb_policy = FunctionPolicy(s -> s.y < 0 ? 1 : -1)
bounds = AdaOPS.IndependentBounds(
    FORollout(lb_policy),
    10.0,
    check_terminal = true,
    consistency_fix_thresh = 1e-5,
)

# ---- 上下界变体 B [已注释]: 完全无信息的常数上下界 (消融用) ----
# 启用时必须同时在 test_ARDESPOT/test_LightDark_ARDESPOT.jl 切到同名变体 B。
# bounds = AdaOPS.IndependentBounds(
#     -10.0,   # 常数下界 = 猜错的一次性惩罚 incorrect_r = -10.0
#      10.0,   # 常数上界 = 猜对的一次性奖励 correct_r = 10.0
#     check_terminal = true,
#     consistency_fix_thresh = 1e-5,
# )

#POMDP solver: just for hyperparameter tuning, not for actual planning
solver = AdaOPSSolver(
    bounds = bounds,
    delta  = 0.3,                                 # R1: 统一为 0.3 (原 0.1); 连续状态下这个值偏粗, R2 扫参时重点看
    grid   = StateGrid(collect(-20.0:1.0:20.0)),  # 状态空间离散化, 供 KLD-sampling 计数
    m_min  = 100,                                 # 自适应粒子数下限
    m_max  = 5000,                                # 自适应粒子数上限
    zeta   = 0.1,                                 # KLD-sampling 的目标散度
    num_b  = 10_000,
    max_depth = mx_depth,                            # 与 POMCPOW 的 max_depth 对齐
    T_max  = 3.0,                                 # 与 POMCPOW 的 max_time = 3.0 一致
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
CSV.write("data_Ada/LD_delta(0.3)_adaops_$(timestamp)_$(mean(results))_R1.csv", df)

# 均值和标准差单独存一个 CSV
summary_df = DataFrame(mean_return = mean(results), std_return = std(results))
CSV.write("data_Ada/LD_delta(0.3)_adaops_$(timestamp)_summary_R1.csv", summary_df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results)), Std: $(std(results))")




#changes 1st round:
#delta=0.1, grid=-20:1:20, m_min=100, m_max=5000, zeta=0.1, num_b=10000
#D=40, T_max=3.0s, 100 runs, 5000 tracking particles, max_steps=100
