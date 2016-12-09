from numpy import *
from intervals import find_best_interval
from collections import defaultdict


OPTIMAL_INTERVALS = [(0, 0.25), (0.5, 0.75)]
VALIDATION_SAMPLE = 8


# needed for section a: generates a label (y) according to the given distribution
def D(x):
    r = random.uniform(0, 1)
    if x < 0.25 or 0.5 < x < 0.75:
        if r > 0.2:
            return x,1
    elif r > 0.9:
        return x,1
    return x,0


# section a: generates m random points and the corresponding label
def draw_pairs(m):
    return sorted(D(random.uniform(0, 1)) for _ in range(m))


# crates arrays of m matching data and labels
def create_intervals(m):
    pairs = draw_pairs(m)
    data = [j for (j, _) in pairs]
    labels = [j for (_, j) in pairs]

    return data, labels


# determines if x belongs to intervals
def find_label(x, intervals):
    if len(intervals) == 0 or intervals[-1][1] < x:
        return 0

    for z in intervals:
        if x < z[1]:
            return int(z[0] < x)
    return 0


# section c: true error calculation
def calculate_true_error(intervals):
    # all continues segments which OPTIMAL and our intervals can agree or disagree on
    segments = sorted([i for z in intervals + OPTIMAL_INTERVALS for i in z] + [1])
    prev, error = 0, 0

    for i in segments:
        # a point in the middle of the segment
        x = average([prev, i])
        opt_label = find_label(x, OPTIMAL_INTERVALS)

        if find_label(x, intervals) != opt_label:
            if opt_label == 1:
            # according to distribution, probability for mistake inside optimal intervals is 0.8
                error += (i - prev) * 0.8
            else:
            # according to distribution, probability for mistake outside optimal intervals is 0.9
                error += (i - prev) * 0.9
        elif opt_label == 1:
            error += (i - prev) * 0.2
        else:
            error += (i - prev) * 0.1

        prev = i

    return error


# section f: cross validation calculation - choose validation size and match
def calculate_cross_val_error(intervals):
    pairs = draw_pairs(VALIDATION_SAMPLE)
    error = 0.0

    for z in pairs:
        error += int(find_label(z[0], intervals) != z[1])

    return error/VALIDATION_SAMPLE

# experiment matches for sections c-f
def experiment(ks, m, T):
    true_error, empirical_error, cross_val_error = defaultdict(int), defaultdict(int), defaultdict(int)

    for i in range(T):
        data, labels = create_intervals(m)

        # array of k pairs of intervals and their error found with ERM, k = no. of intervals
        erm_result = [find_best_interval(data, labels, k) for k in ks]
        intervals = [k for (k, _) in erm_result]
        best_error = [k for (_, k) in erm_result]

        # sums T errors for every kind of error, each is an array of all k, no. of intervals
        true_error = [true_error[j] + calculate_true_error(intervals[j]) for j in range(len(ks))]
        empirical_error = [empirical_error[j] + best_error[j]/float(m) for j in range(len(ks))]
        cross_val_error = [cross_val_error[j] + calculate_cross_val_error(intervals[j]) for j in range(len(ks))]

    # each error averaged over T runs
    return [e/T for e in true_error], [e/T for e in empirical_error],  [e/T for e in cross_val_error]


# section c: true & empirical error as function of m, sample size
def errors_per_sample():
    errors = [(experiment([2], m, 100)) for m in range(10, 101, 5)]
    return [j for (j, _, _) in errors], [j for (_, j, _) in errors]

# sections d,e: true & empirical error as function of k, no. of intervals
def error_per_k():
    errors = experiment(range(1, 21), 50, 100)
    return errors[0], errors[1]


# print experiment(range(1, 21), 42, 1)[2]
# print error_per_k()
