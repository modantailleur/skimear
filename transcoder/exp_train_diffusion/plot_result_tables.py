import numpy as np
import matplotlib.pyplot as plt
import main_doce_diffusion_training as main_doce_diffusion_training
from scipy import stats
import pandas as pd

"""
This file allows the plot of training curve. Please uncomment the row of
the model you want to plot and modify the parameters as you want.
Note that the plan must be specified explicitely (hybridts, cnn or ts).
"""

# Default evalset, can be overwritten in __main__
evalset = 'ljspeech'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select evalset for the training plot.')
    parser.add_argument('--evalset', type=str, default='ljspeech', help='Evaluation dataset (e.g., ljspeech, tau, librispeech)')
    args = parser.parse_args()
    evalset = args.evalset  # Override the default evalset

#hybrid ts
plan_baseline="baseline"
selector_oracle = {"method":"oracle", "step":"metric", 
                   "tho_type":"fast", "evalset":evalset}
selector_pinv = {"method":"pinv", "step":"metric", 
                   "tho_type":"fast", "evalset":evalset}
selector_random = {"method":"random", "step":"metric", 
                   "tho_type":"fast", "evalset":evalset}
plan_diff="evaldiff"
plan_diffwave="evaldiffwave"
selector_diff_tau = {"method":"diffusion", "step":"metric", 
                   "tho_type":"fast", "dataset":"tau", "evalset":evalset,
                   "learning_rate":-4, "epoch":40, "schedule":"DDPM", "diff_steps":1000, "seed":[71]}
selector_diff_libri = {"method":"diffusion", "step":"metric", 
                   "tho_type":"fast", "dataset":"librispeech", "evalset":evalset,
                   "learning_rate":-4, "epoch":40, "schedule":"DDPM", "diff_steps":1000, "seed":[71]}
selector_diff_lj = {"method":"diffusion", "step":"metric", 
                   "tho_type":"fast", "dataset":"ljspeech", "evalset":evalset,
                   "learning_rate":-4, "epoch":70, "schedule":"DDPM", "diff_steps":1000, "seed":[71]}
selector_diffwave_lj = {"method":"diffwave", "step":"metric", 
                   "tho_type":"fast", "dataset":"ljspeech", "evalset":evalset,
                   "learning_rate":-4, "epoch":70, "schedule":"DDPM", "diff_steps":1000, "seed":[71]}
# selector_diff_tau = {"method":"diffusion", "step":"metric", 
#                    "tho_type":"fast", "dataset":"tau", "evalset":evalset,
#                    "learning_rate":-4, "epoch":40, "schedule":"DDPM", "diff_steps":1000}
# selector_diff_libri = {"method":"diffusion", "step":"metric", 
#                    "tho_type":"fast", "dataset":"librispeech", "evalset":evalset,
#                    "learning_rate":-4, "epoch":40, "schedule":"DDPM", "diff_steps":1000}
# selector_diff_lj = {"method":"diffusion", "step":"metric", 
#                    "tho_type":"fast", "dataset":"ljspeech", "evalset":evalset,
#                    "learning_rate":-4, "epoch":70, "schedule":"DDPM", "diff_steps":1000}
selectors = [selector_oracle, selector_pinv, selector_random, selector_diff_tau, selector_diff_libri, selector_diff_lj, selector_diffwave_lj]
plans = [plan_baseline, plan_baseline, plan_baseline, plan_diff, plan_diff, plan_diff, plan_diffwave]
names = ['oracle', 'pinv', 'random', 'diff_tau', 'diff_libri', 'diff_lj', 'diffwave']

# selectors = [selector_oracle, selector_pinv, selector_diff_tau, selector_diff_libri, selector_diff_lj]
# plans = [plan_baseline, plan_baseline, plan_diff, plan_diff, plan_diff]
# names = ['oracle', 'pinv', 'diff_tau', 'diff_libri', 'diff_lj']

# selectors = [selector_pinv, selector_diff_tau, selector_diff_libri]
# plans = [plan_baseline, plan_diff, plan_diff]
# names = ['pinv', 'diff_tau', 'diff_libri']

# selectors = [selector_pinv, selector_random, selector_diff_tau]
# plans = [plan_baseline, plan_baseline , plan_diff]
# names = ['pinv', 'random', 'diff_tau']

stois = []
wer_whisper = []
wer_w2v2 = []
wer_spt = []
wer_ct = []
# wer_cr = []

for selector, plan, name in zip(selectors, plans, names):
    (data, sett, header) = main_doce_diffusion_training.experiment.get_output(
      output = 'stoi',
      selector = selector,
      path = "metric",
      plan = plan
      )
    stois.append(data)
    (data, sett, header) = main_doce_diffusion_training.experiment.get_output(
      output = 'wer_w2v2_gomin',
      selector = selector,
      path = "metric",
      plan = plan
      )
    # Replace values above 1 with 1
    data = np.array(data)
    data = np.where(data > 1, 1, data)
    wer_w2v2.append(data)
    (data, sett, header) = main_doce_diffusion_training.experiment.get_output(
      output = 'wer_whisper_gomin',
      selector = selector,
      path = "metric",
      plan = plan
      )
    # Replace values above 1 with 1
    data = np.array(data)
    data = np.where(data > 1, 1, data)
    wer_whisper.append(data)
    (data, sett, header) = main_doce_diffusion_training.experiment.get_output(
      output = 'wer_spt_gomin',
      selector = selector,
      path = "metric",
      plan = plan
      )
    # Replace values above 1 with 1
    data = np.array(data)
    data = np.where(data > 1, 1, data)
    wer_spt.append(data)

    (data, sett, header) = main_doce_diffusion_training.experiment.get_output(
      output = 'wer_ct_gomin',
      selector = selector,
      path = "metric",
      plan = plan
      )
    # Replace values above 1 with 1
    data = np.array(data)
    data = np.where(data > 1, 1, data)
    wer_ct.append(data)

    # (data, sett, header) = main_doce_diffusion_training.experiment.get_output(
    #   output = 'wer_cr_gomin',
    #   selector = selector,
    #   path = "metric",
    #   plan = plan
    #   )
    # # Replace values above 1 with 1
    # data = np.array(data)
    # data = np.where(data > 1, 1, data)
    # wer_cr.append(data)

# Perform pairwise t-tests and store results
scores = [stois, wer_w2v2, wer_spt, wer_whisper, wer_ct]

score_types = ["upper", "lower", "lower", "lower", "lower"]
means = []
stds = []
all_data = []
put_in_bolds = []
for _, (arrays, score_type) in enumerate(zip(scores, score_types)):
    p_values = np.ones((len(arrays), len(arrays)))  # Initialize a 10x10 matrix to store p-values
    mean_arrays = [np.mean(array) for array in arrays]
    stds_arrays = [np.std(array) for array in arrays]
    shape_arrays = [np.array(array).shape for array in arrays]
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            array_i = np.array(arrays[i]).flatten()
            array_j = np.array(arrays[j]).flatten()
            t_stat, p_value = stats.ttest_ind(array_i, array_j)
            p_values[i, j] = p_value
            p_values[j, i] = p_value

    p_values_significant = p_values < 0.05

    # # Find the array with the lowest mean
    # min_mean_index = np.argmin(mean_arrays)
    # min_mean_array_name = names[min_mean_index]
    # min_mean_p_values = p_values[min_mean_index, :]

    # # Determine if the lowest mean is significantly lower than others
    # significant_comparisons = min_mean_p_values < 0.05

    if score_type == "lower":
        idx = np.argmin(mean_arrays)
    else:
        idx = np.argmax(mean_arrays)
    put_in_bold = np.logical_not(p_values_significant[idx, :])
    means.append(mean_arrays)
    stds.append(stds_arrays)
    put_in_bolds.append(put_in_bold)

    for m in mean_arrays:
        if np.isnan(m):
            pass
            # Do something if m is NaN
            # print(print(arrays[2]))

# Prepare data for DataFrame
# columns = ["name", "stoi", "wer_w2v2_gomin", "wer_whisper_gomin", "wer_spt_gomin", "wer_ct_gomin", "wer_cr_gomin",
#            "bold_stoi", "bold_wer_w2v2_gomin", "bold_wer_whisper_gomin", "bold_wer_spt_gomin", "bold_wer_ct_gomin", "bold_wer_cr_gomin"]
columns = ["name", "w2v2", "whisper", "spt", "crdnn", "ASR combined",
           "bold_stoi", "bold_wer_w2v2_gomin", "bold_wer_whisper_gomin", "bold_wer_spt_gomin", "bold_wer_ct_gomin"]
data = []

# for i, name in enumerate(names):
#     row = [
#         name,
#         means[0][i], means[1][i], means[2][i], means[3][i], means[4][i], means[5][i],
#         put_in_bolds[0][i],  put_in_bolds[1][i], put_in_bolds[2][i], put_in_bolds[3][i], put_in_bolds[4][i], put_in_bolds[5][i]
#     ]
#     data.append(row)

# def find_shape(list):
#     shape = [len(list)]
#     for elem1 in list:
#         if len(elem1) != 0:
#             shape.append(len(elem1))
#             for elem2 in elem1:
#                 if len(elem2) != 0:
#                     shape.append(len(elem2))
#                     for elem3 in elem2:
#                         if len(elem3) != 0:
#                           shape.append(len(elem3))
#                           for elem4 in elem3:
#                               if len(elem4) != 0:
#                                 shape.append(len(elem4))
#                                 break
#                           break
#                     break
#             break
#     return tuple(shape)

def find_shape(lst):
    shape = []
    while isinstance(lst, (list, np.ndarray)) and len(lst) > 0:
        shape.append(len(lst))
        lst = lst[0]  # Move to the next level of the list
    return tuple(shape)

def list_to_array(lst):
    shape = find_shape(lst)
    # Create an array filled with NaNs of the desired shape
    array = np.full(shape, np.nan)
    
    def fill_array(sublist, indices):
        if isinstance(sublist, list):
            if len(sublist) != 0:
              for i, item in enumerate(sublist):
                  fill_array(item, indices + (i,))
        else:
            if len(sublist) != 0:
              # Use indices to place the item in the array
              array[indices] = sublist
    
    fill_array(lst, ())
    return array

scores_reshaped = list_to_array(scores)
for i, name in enumerate(names):
    row = [name]
    row += [f"{100*means[j][i]:.2f}±{100*stds[j][i]:.2f}" for j in range(1, len(means))]
    to_avg = np.array(scores_reshaped[1:], dtype=float)[:,i].flatten()
    row += [f"{100*np.mean(to_avg):.1f}±{100*np.std(to_avg):.1f}"]
    row += [put_in_bolds[j][i] for j in range(len(put_in_bolds))]
    #     name,
    #     means[0][i], means[1][i], means[2][i], means[3][i], means[4][i], means[5][i],
    #     put_in_bolds[0][i],  put_in_bolds[1][i], put_in_bolds[2][i], put_in_bolds[3][i], put_in_bolds[4][i], put_in_bolds[5][i]
    # ]
    data.append(row)

print(len(data))
df = pd.DataFrame(data, columns=columns)

df = df.loc[:, ~df.columns.str.startswith('bold')]
# df = df[['name', 'wer_w2v2_gomin', 'bold_wer_w2v2_gomin']]
print(df)
# print(df[["wer_ct_gomin", "wer_cr_gomin"]])

# Save DataFrame as a LaTeX table
latex_table = df.to_latex(index=False)
with open(f"figures/{evalset}_table.tex", "w") as f:
    f.write(latex_table)
