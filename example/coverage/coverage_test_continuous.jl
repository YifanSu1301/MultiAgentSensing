using PyPlot

using SubmodularMaximization
using Distributions
using LinearAlgebra
using Random


pygui(false)
fig_path = "./fig/coverage_test"
mkpath(fig_path)

num_agents = 5
num_sensors = 5
nominal_area = 2.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = sensor_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

agents = generate_agents(agent_specification, num_agents)

f(x) = mean_area_coverage(x, 100)
problem = ExplicitPartitionProblem(f, agents)

###### Continuous Greedy Algorithm Test #######



# Define the number of players and items
function test_continuous()

    # Initialize y matrix
    y = zeros(num_agents, num_sensors)

    # Initialize t and delta
    t = 0.0
    delta = 1.0 / (num_agents * num_sensors)*10


    # Define a function to estimate the expected marginal profit of player i from item j
    function expected_marginal_profit(i, j, y, w)
        # Generate a random set Ri containing each item j independently with probability yij(t)
        R = [x for x = 1:num_agents if x != j && rand() < y[i, x]]
        g(j) = problem.partition_matroid[j].sensors[i]
        R = map(g, R)
        newR = copy(R)
        push!(newR, problem.partition_matroid[j].sensors[i])

        # Compute the expected marginal profit of player i from item j
        wi_Ri_j = w(newR)
        wi_Ri = w(R)
        return wi_Ri_j - wi_Ri
    end

    # Initialize the distribution of all mn actions
    # action_distribution = zeros(n_players, n_items)

    # Run the algorithm
    while t < 1
        # Estimate the expected marginal profits for all players and items
        
        ω = zeros(num_sensors, num_agents)
        for i = 1:num_sensors
            # if(t==0) break end
            for j = 1:num_agents
                
                for k = 1:(num_sensors * num_agents)
                    ω[i, j] += expected_marginal_profit(i, j, y, problem.objective)
                    # println("Loop4\n");
                end
                ω[i, j] /= (num_sensors * num_agents)
                # println("Loop3\n");
            end
            # println("Loop2\n");
        end
        # Update y matrix
        for j = 1:num_agents
            i_star = argmax(ω[:, j])
            y[i_star, j] += delta
            
        end
        # Increment t
        t += delta
        println("Loop1: $t");
    end

    # println("OutLoop1\n");

    # Print the action distribution matrix
    println("Action distribution matrix:")
    println(y)
    y

end

y = test_continuous()
solution = solve_continuous(problem, y)
# solution = solve_sequential(problem)
# solution = evaluate_solution(problem, selection)
# solve_continuous(problem)

function evaluate_solver(sol,name)
  println("$name solver running")
  @time solution = sol

  figure()
  xlim([0, 1])
  ylim([0, 1])
  colors = generate_colors(agents)
  visualize_agents(agents, colors)
  visualize_solution(problem, solution, colors)

  gca().set_aspect("equal")
  savefig("$(fig_path)/$(to_file(name)).png", pad_inches=0.00, bbox_inches="tight")

  coverage = solution.value

  title("$name Solver ($coverage)")

  @show coverage
end

evaluate_solver(solution, "Continuous")



@show mean_weight(problem)
@show total_weight(problem)



