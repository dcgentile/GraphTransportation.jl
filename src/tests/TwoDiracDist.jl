# a Monte Carlo computation of the distance between dirac masses

"""
    approximate_distance_by_integral(num_samples=10000)

Description of the function.

#TODO
"""
function approximate_distance_by_integral(num_samples=10000)
    f(x) = 1/√(√((1-x)*(1 + x)))
    vals = [f(2*rand() - 1) for _ in 1:num_samples]
    return √2 * sum(vals) / num_samples
end
