using SubmodularMaximization
using POMDPs
using MCTS

using Statistics
using Profile
using ProfileView

default_steps = 2
grid_size = 10
horizon = 2
iterations = 100

function run_test(steps)
  println("Running test: ", steps, " steps")
  grid = Grid(grid_size, grid_size)
  sensor = RangingSensor(0.5^2, 0.1^2)

  robot_state = random_state(grid)
  target_state = random_state(grid)

  target_states = Array{State}(undef, steps)
  target_states[1] = target_state

  robot_states = Array{State}(undef, steps)
  robot_states[1] = robot_state

  histogram_filter = Filter(grid, target_state)

  problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                             [histogram_filter])

  time = @elapsed for ii = 2:steps
    println("Step: ", ii)

    # Before the target moves and the robot receives a measurement, execute robot
    # dynamics
    robot_state = solve_single_robot(problem, robot_state,
                                     n_iterations = iterations)
    robot_states[ii] = robot_state

    # Then update the target and sample the observation
    target_state = target_dynamics(grid, target_states[ii-1])
    target_states[ii] = target_state

    range_observation = generate_observation(sensor, robot_state, target_state)

    # After updating states, update the filter (in place for now)
    process_update!(histogram_filter, transition_matrix(grid))
    measurement_update!(histogram_filter, robot_state, get_states(grid), sensor,
                        range_observation)
  end

  println("Test duration: ", time, " seconds")
  println("  ", time / (steps - 1), " seconds per step")
end

Profile.init(n=1000000)
#@profview run_test(2)
@profile run_test(2)
#run_test(2)

Profile.clear()
ProfileView.closeall()
Profile.init(n=1000000)
#@profview run_test(default_steps)
#run_test(default_steps)
@profile run_test(default_steps)
Profile.print(mincount=100)

nothing
