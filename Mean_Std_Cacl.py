import statistics

ade_data = [1.4, 1.36, 1.38, 1.38, 1.42, 1.42, 1.39, 1.39, 1.39]
fde_data = [2.96, 2.86, 2.96, 2.93, 3, 3.01, 2.94, 2.93, 2.94]

print('ade_mean', statistics.mean(ade_data))
print('fde_mean', statistics.mean(fde_data))

print('ade_std', statistics.stdev(ade_data))
print('fde_std', statistics.stdev(fde_data))
