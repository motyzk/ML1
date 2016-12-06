from numpy import *
from intervals import find_best_interval
from matplotlib.pyplot import *


OPTIMAL_INTERVALS = [(0, 0.25), (0.5, 0.75)]


def D(i):
    x = random.uniform(0, 1)
    if i < 0.25 or 0.5 < i < 0.75:
        if x > 0.2:
            return i,1
    elif x > 0.9:
        return i,1
    return i,0


def draw_pairs(m):
    return sorted(D(random.uniform(0, 1)) for _ in range(m))


def find_label(x, intervals):
    if len(intervals) == 0 or intervals[-1][1] < x:
        return 0

    for z in intervals:
        if x < z[1]:
            return int(z[0] < x)
    return 0


def calculate_error(intervals):
    segments = sorted([i for z in intervals + OPTIMAL_INTERVALS for i in z] + [1])
    prev, error = 0, 0

    for i in segments:
        x = average([prev, i])
        if find_label(x, intervals) != find_label(x, OPTIMAL_INTERVALS):
            error += i - prev
        prev = i
    return error


def experiment(k, m, T):
    empirical_error, true_error = 0, 0

    for i in range(T):
        pairs = draw_pairs(m)
        intervals, best_error = find_best_interval([j for (j, _) in pairs], [j for (_, j) in pairs], k)
        empirical_error += float(best_error) / m
        true_error += calculate_error(intervals)

    return true_error/T, empirical_error/T


def error_range():
    errors = [(experiment(2, m, 100)) for m in range(10, 101, 5)]
    return [k for (k, _) in errors], [k for (_, k) in errors]
