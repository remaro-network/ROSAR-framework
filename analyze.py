import argparse
import numpy as np
import pandas as pd
import random
import re

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 12})

STAT_PATHS = {
    'P1': {
        'original': './statistics_<T1>.csv',
        'pgd-retrained': './statistics_<T2>.csv',
        'patch-retrained': './statistics_<T3>.csv'
    },
    'P2': {
        'original': './statistics_<T4>.csv',
        'pgd-retrained': './statistics_<T5>.csv',
        'patch-retrained': './statistics_<T6>.csv'
    }
}

LABS = ['Original', 'PGD-retrained', 'PATCH-retrained']

def P1_estimate_bound(bound_results):
    not_found_bound = 0.00
    found_bound = 1.00
    for bound_result in bound_results:
        if bound_result[0] == 'FOUND' and found_bound > bound_result[1]:
            found_bound = bound_result[1]
        if bound_result[0] == 'NOT_FOUND' and not_found_bound < bound_result[1]:
            not_found_bound = bound_result[1]

    return (found_bound + not_found_bound) / 2.0

def P2_estimate_bound(bound_results):
    not_found_bound = 1.00
    found_bound = 0.60
    for bound_result in bound_results:
        if bound_result[0] == 'FOUND' and found_bound < bound_result[1]:
            found_bound = bound_result[1]
        if bound_result[0] == 'NOT_FOUND' and not_found_bound > bound_result[1]:
            not_found_bound = bound_result[1]

    return (found_bound + not_found_bound) / 2.0

def extract_P1_bounds(input_path):
    stat_df = pd.read_csv(input_path, header=None)
    stat_df.columns = ['Result', 'img_path', 'adv_bound', 'vnn_name']
    bbox_index_pattern = '.*perturbed_bbox_([0-9]+)_delta.*'
    stat_df['bbox_index'] = stat_df.apply(lambda row: int(re.search(bbox_index_pattern, row['vnn_name'], re.IGNORECASE).group(1)), axis=1)
    list1 = stat_df['bbox_index'].tolist()
    list2 = [int(name[79]) for name in stat_df['vnn_name'].tolist()]
    # print(list(zip(list1, list2)))
    for ind1, ind2 in zip(list1, list2):
        if ind1 != ind2:
            print('error:', ind1, ind2)

    bounds = []
    img_paths = set(stat_df['img_path'].tolist())
    for img_path in img_paths:
        img_df = stat_df[stat_df['img_path'] == img_path]
        # print(img_df)
        bbox_ids = set(img_df['bbox_index'].tolist())
        for bbox_id in bbox_ids:
            img_bbox_df = img_df[img_df['bbox_index'] == bbox_id]
            # print(img_bbox_df)

            results = img_bbox_df['Result'].tolist()
            adv_bounds = img_bbox_df['adv_bound'].tolist()
            img_ID = img_path[-7:-4]
            bbox_ID = bbox_id
            coupled_result = list(zip(results, adv_bounds))
            est_bound = P1_estimate_bound(coupled_result)
            # print(img_ID, bbox_ID, coupled_result, '->', est_bound)

            bounds.append(est_bound)

    return bounds

def extract_P2_bounds(input_path):
    stat_df = pd.read_csv(input_path, header=None)
    stat_df.columns = ['Result', 'img_path', 'adv_bound', 'vnn_name']
    bbox_index_pattern = '.*black_lines_([0-9]+)_min_delta.*'
    stat_df['bbox_index'] = stat_df.apply(lambda row: int(re.search(bbox_index_pattern, row['vnn_name'], re.IGNORECASE).group(1)), axis=1)
    list1 = stat_df['bbox_index'].tolist()
    list2 = [int(name[76]) for name in stat_df['vnn_name'].tolist()]
    # print(list(zip(list1, list2)))
    for ind1, ind2 in zip(list1, list2):
        if ind1 != ind2:
            print('error:', ind1, ind2)

    bounds = []
    img_paths = set(stat_df['img_path'].tolist())
    for img_path in img_paths:
        img_df = stat_df[stat_df['img_path'] == img_path]
        # print(img_df)
        bbox_ids = set(img_df['bbox_index'].tolist())
        for bbox_id in bbox_ids:
            img_bbox_df = img_df[img_df['bbox_index'] == bbox_id]
            # print(img_bbox_df)

            results = img_bbox_df['Result'].tolist()
            adv_bounds = img_bbox_df['adv_bound'].tolist()
            img_ID = img_path[-7:-4]
            bbox_ID = bbox_id
            coupled_result = list(zip(results, adv_bounds))
            est_bound = P2_estimate_bound(coupled_result)
            # print(img_ID, bbox_ID, coupled_result, '->', est_bound)

            bounds.append(est_bound)

    return bounds

def draw_brace(ax, xspan, text):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution//2+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = 0.070 + ymin + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)

    ax.text((xmax+xmin)/2., 0.07 + ymin+.07*yspan, text, ha='center', va='bottom')

def create_raincloud_plots():
    P1_bounds = [extract_P1_bounds(STAT_PATHS['P1']['original']),
        extract_P1_bounds(STAT_PATHS['P1']['pgd-retrained']),
        extract_P1_bounds(STAT_PATHS['P1']['patch-retrained'])
    ]
    min_len_P1 = min(len(P1_bounds[0]), len(P1_bounds[1]), len(P1_bounds[2]))
    P1_bounds = [random.sample(P1_bounds[0], min_len_P1),
                 random.sample(P1_bounds[1], min_len_P1),
                 random.sample(P1_bounds[2], min_len_P1)]

    P2_bounds = [
        extract_P2_bounds(STAT_PATHS['P2']['original']),
        extract_P2_bounds(STAT_PATHS['P2']['pgd-retrained']),
        extract_P2_bounds(STAT_PATHS['P2']['patch-retrained'])
    ]
    min_len_P2 = min(len(P2_bounds[0]), len(P2_bounds[1]), len(P2_bounds[2]))
    P2_bounds = [random.sample(P2_bounds[0], min_len_P2),
                 random.sample(P2_bounds[1], min_len_P2),
                 random.sample(P2_bounds[2], min_len_P2)]


    print('Mean values for P1:', np.mean(P1_bounds, axis=1))
    print('Median values for P1:', np.median(P1_bounds, axis=1))
    print('Mean values for P1:', np.mean(P2_bounds, axis=1))
    print('Median values for P1:', np.median(P2_bounds, axis=1))
    
    fig, ax1 = plt.subplots(figsize=(16, 4))

    ax2 = ax1.twinx()

    # Create a list of colors for the scatter plots based on the number of features you have
    scatter_colors = ['#4B0082', '#FA8072', '#808000', '#4B0082', '#FA8072', '#808000']

    # Scatterplot data
    for idx, features in enumerate(P1_bounds):
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + 1.125)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        ax1.scatter(y, features, s=.5, alpha=0.4, c=scatter_colors[idx])
        # features.sort()
        # plt.scatter(features, range(len(features)), s=2.0, c=scatter_colors[idx])

    # Create a list of colors for the boxplots based on the number of features you have
    boxplots_colors = ['#4B0082', '#FA8072', '#808000', '#4B0082', '#FA8072', '#808000']

    # Boxplot data
    bp = ax1.boxplot(P1_bounds,                      
                     patch_artist = True, vert = True, showmeans=True, showfliers=False, positions=[1.125, 2.125, 3.125], widths=[0.125, 0.125, 0.125],
                     whiskerprops=dict(linewidth=1.5, color='#708090'), capprops=dict(linewidth=1.5, color='#708090'), boxprops=dict(linewidth=1.5, fill=None, color='#708090'), medianprops=dict(linewidth=1.5, color='#4682B4'), meanprops=dict(markerfacecolor='#4682B4',markeredgecolor='#4682B4',markersize=7.5))


    # Change to the desired color and add transparency
    for idx, patch in enumerate(bp['boxes']):
        patch.set_facecolor(boxplots_colors[idx])
        patch.get_path().vertices[:, 0] = np.clip(patch.get_path().vertices[:, 0], idx+1, idx+2)
        # patch.set_alpha(0.4)

    # Create a list of colors for the violin plots based on the number of features you have
    violin_colors = ['#4B0082', '#FA8072', '#808000', '#4B0082', '#FA8072', '#808000']

    # Violinplot data
    vp = ax1.violinplot(P1_bounds, points=min_len_P1, 
                showmeans=False, showextrema=False, showmedians=False, vert=True, positions=[1, 2, 3])

    for idx, b in enumerate(vp['bodies']):
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx, idx+1)
        # Change to the desired color
        b.set_color(violin_colors[idx])
        b.set_edgecolor('black')
        b.set_linewidth(1.5)


    # Create a list of colors for the scatter plots based on the number of features you have
    scatter_colors = ['#4B0082', '#FA8072', '#808000']

    # Scatterplot data
    for idx, features in enumerate(P2_bounds):
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + 1.075 + 3.05)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        ax2.scatter(y, features, s=.5, alpha=0.4, c=scatter_colors[idx])
        # features.sort()
        # plt.scatter(features, range(len(features)), s=2.0, c=scatter_colors[idx])

    # Create a list of colors for the boxplots based on the number of features you have
    boxplots_colors = ['#4B0082', '#FA8072', '#808000']

    # Boxplot data
    bp = ax2.boxplot(P2_bounds, 
                     patch_artist = True, vert = True, showmeans=True, showfliers=False, positions=[4.125, 5.125, 6.125], widths=[0.125, 0.125, 0.125],
                     whiskerprops=dict(linewidth=1.5, color='#708090'), capprops=dict(linewidth=1.5, color='#708090'), boxprops=dict(linewidth=1.5, fill=None, color='#708090'), medianprops=dict(linewidth=1.5, color='#4682B4'), meanprops=dict(markerfacecolor='#4682B4',markeredgecolor='#4682B4',markersize=7.5))

    # Change to the desired color and add transparency
    for idx, patch in enumerate(bp['boxes']):
        patch.set_facecolor(boxplots_colors[idx])
        patch.get_path().vertices[:, 0] = np.clip(patch.get_path().vertices[:, 0], idx+4, idx+5)
        # patch.set_alpha(0.8)
    
    # for idx, median in enumerate(bp['medians']):
    #     median.get_path().vertices[:, 0] = np.clip(median.get_path().vertices[:, 0], idx+4, idx+5)

    
    # for idx, cap in enumerate(bp['caps']):
    #     cap.get_path().vertices[:, 0] = np.clip(cap.get_path().vertices[:, 0], idx+4, idx+5)

    # Create a list of colors for the violin plots based on the number of features you have
    violin_colors = ['#4B0082', '#FA8072', '#808000']

    # Violinplot data
    vp = ax2.violinplot(P2_bounds, points=min_len_P2, 
                showmeans=False, showextrema=False, showmedians=False, vert=True, positions=[4, 5, 6])

    for idx, b in enumerate(vp['bodies']):
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+3, idx+4)
        # Change to the desired color
        b.set_color(violin_colors[idx])
        b.set_edgecolor('black')
        b.set_linewidth(1.5)

    labels = []
    labels.append((mpatches.Patch(color='#4B0082'), 'Original'))
    labels.append((mpatches.Patch(color='#FA8072'), 'PGD-retrained'))
    labels.append((mpatches.Patch(color='#808000'), 'Patch-retrained'))

    draw_brace(ax1, (0.75, 3.25), r'$\mathcal{P}_1$')
    draw_brace(ax1, (3.75, 6.25), r'$\mathcal{P}_2$')

    ax1.set_xticks(np.arange(1,7,1), ['Original', 'PGD-retrained', 'Patch-retrained', 'Original', 'PGD-retrained', 'Patch-retrained'])  # Set text labels.
    # ax1.set_ylabel('Robustness bounds')
    ax1.set_ylim([-0.0015, 0.08])
    ax2.set_ylim([0.59, 1.075])
    ax1.set_yticks([0.00, 0.02, 0.04, 0.06])
    ax2.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    # ax2.set_xticks(np.arange(1,4,1), ['Original', 'PGD-retrained', 'Patch-retrained'])  # Set text labels.
    # ax2.set_xlabel('Robustness bounds')
    # plt.legend(*zip(*labels), prop={'size': 12}, bbox_to_anchor=(0.0,1.2))
    plt.savefig("raincloud.svg")
    plt.show()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('P1_original', type=str, help='path to the P1 statistics file generated using the original model')
    parser.add_argument('P1_pgd_retraing', type=str, help='path to the P1 statistics file generated using the PGD-retrained model model')
    parser.add_argument('P1_patch_retraing', type=str, help='path to the P1 statistics file generated using the patch-retrained model model')
    parser.add_argument('P2_original', type=str, help='path to the P2 statistics file generated using the original model')
    parser.add_argument('P2_pgd_retraing', type=str, help='path to the P2 statistics file generated using the PGD-retrained model model')
    parser.add_argument('P2_patch_retraing', type=str, help='path to the P2 statistics file generated using the patch-retrained model model')
    opt = parser.parse_args()
    print('CMD Arguments:', opt)
    return opt

def main(opt):
    STAT_PATHS['P1']['original'] = opt.P1_original
    STAT_PATHS['P1']['pgd-retrained'] = opt.P1_pgd_retraing
    STAT_PATHS['P1']['patch-retrained'] = opt.P1_patch_retraing
    STAT_PATHS['P2']['original'] = opt.P2_original
    STAT_PATHS['P2']['pgd-retrained'] = opt.P2_pgd_retraing
    STAT_PATHS['P2']['patch-retrained'] = opt.P2_patch_retraing
    # print(STAT_PATHS)

    create_raincloud_plots()

if __name__=='__main__':
    opt = parse_opt()
    main(opt)






