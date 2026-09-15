using POMCGraphSearch
using POMDPs
using LaserTag
using CSV, DataFrames
using Random 

Random.seed!(1)

rng = MersenneTwister(7)
pomdp = gen_lasertag(rng=rng, robot_position_known=false)

action_space = actions(pomdp)

# 参数设置
nb_runs = 100
max_steps = 100

# 存储数据 - 直接存数组
data = DataFrame(
    a = Int[],
    sp = Vector{Float32}[],
    o = Vector{Float32}[],
    r = Float64[]
)

results = Float64[]

for run in 1:nb_runs
    s = rand(rng, initialstate(pomdp))
    total_reward = 0.0
    
    for step in 1:max_steps
        a = rand(rng, action_space)
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        o_vec = convert_o(Vector{Float32}, o, pomdp)
        sp_vec = convert_s(Vector{Float32}, sp, pomdp)
        
        
        push!(data, (a, sp_vec, o_vec, r))
        
        total_reward += r
        s = sp
        
        if isterminal(pomdp, sp)
            break
        end
    end
    
    push!(results, total_reward)
end

# 保存到CSV
CSV.write("data_LaserTag.csv", data)

println("Total transitions: $(nrow(data))")
println("Average return: $(mean(results))")