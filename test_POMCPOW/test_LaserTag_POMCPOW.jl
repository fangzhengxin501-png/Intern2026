using POMCPOW
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using LaserTag
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


Random.seed!(1)

#POMDP problem
# 与 test_POMCGS/experiments_online/test_LaserTag.jl 保持完全一致的问题实例,
# 这样 POMCPOW 和 POMCGS 的结果可以直接对比。
rng_env = MersenneTwister(7)
pomdp = gen_lasertag(rng = rng_env, robot_position_known = false)
#state: discrete (robot Coord, opponent Coord)
#action: discrete (8 个: 4 个移动 + measure 等)
#observation: discrete (DESPOTEmu / DMeas, 离散激光读数)

mx_depth = 40 #defult 10, try fixed 40
mx_steps = 100
nb_runs   = 100
n_particles = 5000

#POMDP solver: just for hyperparameter tuning, not for actual planning
# 超参数取自 JuliaPOMDP/POMCPOW.jl 的 README 默认值, 只在"离散问题"相关的项上做调整,
# 并与本目录下 test_RS15_POMCPOW.jl 的调参口径保持一致。
solver = POMCPOWSolver(
    criterion       = MaxUCB(1.0),  # R1: 回到 POMCPOW.jl README 默认值 1.0 (原为 20.0)
    k_observation   = 10.0,            # README default k_observation = 10, 渐进加宽参数: 观测扩展速率
    alpha_observation = 0.5,         # README default alpha_observation = 0.5 (离散观测保持默认)
    enable_action_pw  = false,       # LaserTag 动作空间离散且很小, 无需对动作做渐进加宽 (README: 离散动作建议关掉)
    max_depth = mx_depth,
    tree_queries = typemax(Int),          # 设得足够大, 让 max_time 成为实际的停止条件
    max_time = 3.0,                     #try fixed 3s
    check_repeat_obs  = true,        #discrete obs, so we can check repeat obs
    check_repeat_act  = true,        #discrete act
    estimate_value = FORollout(RandomPolicy(pomdp, rng = MersenneTwister(3))), #lower bound 一致性
    rng = MersenneTwister(1)
)

# ---- 叶节点价值估计 (= 下界) 说明 ----
# POMCPOW 没有"上下界", 但 estimate_value 起的作用和 AdaOPS/ARDESPOT 的**下界**完全一样:
# 都是在叶节点上用一个策略 rollout 出一个价值估计。
#
# 2026-09-17 修正: 原来这里没设 estimate_value, 靠 POMCPOW 的默认随机 rollout。
# 对**随机策略**而言, 默认随机 rollout 与 Ada/ARDESPOT 的
# FORollout/DefaultPolicyLB(RandomPolicy(...)) 在分布上是等价的 (随机策略既不看状态
# 也不看观测), 所以先验信息量本来就是对齐的。但默认值有三个问题:
#   (1) 它的 rng 是 POMCPOW 内部构造的, 不是本项目统一的 MersenneTwister(3),
#       是种子方案里唯一一个不可控的随机源;
#   (2) 依赖包默认值 —— POMCPOW 升版换掉默认 estimator, 只有这个基线会静默变;
#   (3) 另外三个 POMCPOW 脚本 (RS11/RS15/LD) 都显式设了, 只有这里例外。
# 因此改成显式写法, 与 test_Ada/test_LaserTag_Ada.jl 和
# test_ARDESPOT/test_LaserTag_ARDESPOT.jl 的启用项逐字对应 (同策略、同种子 3)。
#
# ---- 下界变体 B [已注释]: 自写的 LaserTag 启发式 ----
# 切到变体 B 时, 三个 LaserTag 文件必须同步换成同一个启发式策略
# (Ada/ARDESPOT 那边是 lt_heuristic, 见各自文件内的注释), 否则对比不公平。
# 不要用 LaserTag.jl 自带的 MoveTowards(): 它的 action() 只在差向量各分量 ∈ {-1,0,1}
# 时才有定义, 否则 DIR_TO_ACTION[...] 抛 KeyError, 且可能返回 6..9 的对角动作。
#
#     estimate_value = FORollout(lt_heuristic),
#
# (FORollout / FOValue 由 POMCPOW 通过 BasicPOMCP re-export, 可直接用。)

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
CSV.write("data_POMCPOW/LaserTag_UCB(1.0)_pomcpow_$(timestamp)_$(mean(results))_R1.csv", df)

# 均值和标准差单独存一个 CSV
summary_df = DataFrame(mean_return = mean(results), std_return = std(results))
CSV.write("data_POMCPOW/LaserTag_UCB(1.0)_pomcpow_$(timestamp)_summary_R1.csv", summary_df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results)), Std: $(std(results))")




#changes 1st round:
#ucb(20), max_depth=40, max_time=3.0, p_time =3s +inf tree_queries
#k_o=10.0, alpha_o=0.5 (POMCPOW.jl README defaults), action PW off (discrete actions)
#100 runs, 5000 particles, max_steps=100
