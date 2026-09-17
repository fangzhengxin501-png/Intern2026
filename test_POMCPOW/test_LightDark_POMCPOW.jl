using POMCPOW
using POMDPModels, POMDPTools, POMDPs, Distributions, StatsBase
using CSV, DataFrames
using Random
using Dates
using ParticleFilters
using Statistics

#现有改动的数据来自POMCPOW附录

Random.seed!(1)

#POMDP problem
pomdp = LightDark1D()


mx_depth = 40 #defult 10, try fixed 40
mx_steps = 100
nb_runs   = 100#
n_particles = 5000

#POMDP solver: just for hyperparameter tuning, not for actual planning
solver = POMCPOWSolver(
    criterion       = MaxUCB(1.0),  # R1: 回到 POMCPOW.jl README 默认值 1.0 (原为 20.0)
    k_observation   = 10.0,            # 渐进加宽参数：观测扩展速率
    alpha_observation = 0.5,         # 对应粗略类比 num_fixed_observations=10（观测聚类越少，这里可适当调小 k/alpha）
    max_depth = mx_depth,
    tree_queries = typemax(Int),          # 设得足够大, 让 max_time 成为实际的停止条件
    max_time = 3.0,                     
    check_repeat_obs  = false,
    estimate_value = FORollout(FunctionPolicy(s -> s.y < 0 ? 1 : -1)), #lower bound 一致性
    rng = MersenneTwister(1)
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
CSV.write("data_POMCPOW/LD_UCB(1.0)_alphao(1_15)_pomcpow_$(timestamp)_$(mean(results))_R1.csv", df)

# 均值和标准差单独存一个 CSV
summary_df = DataFrame(mean_return = mean(results), std_return = std(results))
CSV.write("data_POMCPOW/LD_UCB(1.0)_alphao(1_15)_pomcpow_$(timestamp)_summary_R1.csv", summary_df)

println("\nTotal return: $(sum(results))")
println("Average return: $(mean(results)), Std: $(std(results))")




#changes 1st round:
#ucb(20), max_depth=40, max_time=3.0, p_time =3s +inf tree_queries
#with offset hyperparameter tuning, 100 runs, 5000 particles, max_steps=100
