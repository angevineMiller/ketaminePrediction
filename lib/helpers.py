import os
import math
import numpy as np
import pandas as pd
from datetime import date
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def exponential(x, a, b, c):
    return a * np.exp(-b*x) + c

def rss_exponential(params, xs, ys):
    a, b, c = params
    sqr_errs = []
    for i, x in enumerate(xs):
        y = ys[i]
        yhat = exponential(x, a, b, c)
        sqr_errs.append((y - yhat)**2)
    return np.sum(sqr_errs)

def rss_linear(params, xs, ys):
    slope, intercept = params
    sqr_errs = []
    for i, x in enumerate(xs):
        y = ys[i]
        yhat = slope * x + intercept
        sqr_errs.append((y - yhat)**2)
    return np.sum(sqr_errs)


def fit_lin_patient_item(d, item_idx):
    item_name = 'phqitem' + str(item_idx)
    max_session = min(d['sessionNumber'].max(), 8)
    init_sess_date = d[d['sessionNumber']==1]['sessionDay'].iloc[0]
    xs = []
    ys = []
    errs = []
    for session_id in np.arange(1, max_session+1):
        d_sess = d[d['sessionNumber']==session_id]
        sess_date = d_sess['sessionDay'].iloc[0]
        day_diff = sess_date - init_sess_date
        item_score = d_sess[item_name].iloc[0]
        if (day_diff < 0): 
            continue
        if np.isnan(item_score):
            continue
        xs.append(day_diff)
        ys.append(item_score)
    res = minimize(rss_linear, x0=[-1, 1], args=(np.array(xs), np.array(ys)))
    slope, intercept = res.x
    rss = res.fun
    return slope, intercept, rss

    
def fit_exp_patient_item(d, item_idx):
    item_name = 'phqitem' + str(item_idx)
    max_session = min(d['sessionNumber'].max(), 8)
    init_sess_date = d[d['sessionNumber']==1]['sessionDay'].iloc[0]
    xs = []
    ys = []
    for session_id in np.arange(1, max_session+1):
        d_sess = d[d['sessionNumber']==session_id]
        sess_date = d_sess['sessionDay'].iloc[0]
        day_diff = sess_date - init_sess_date
        item_score = d_sess[item_name].iloc[0]
        if (day_diff < 0): 
            continue
        if np.isnan(item_score):
            continue
        xs.append(day_diff)
        ys.append(item_score)
    res = minimize(rss_exponential, x0=[1, 0, -1], args=(np.array(xs), np.array(ys)))
    a, b, c = res.x
#     rss = rss_exponential((a, b, c), xs, ys)
    rss = res.fun
    return a, b, c, rss


def get_final_session(d_patient):
    max_sess_number = d_patient['sessionNumber'].max() # overall max session
    if max_sess_number < 8:
        return max_sess_number 
    else:
        # Set final session to the lowest existing session number >= 8
        sess_above_8 = d_patient[d_patient['sessionNumber'] >= 8]['sessionNumber'].values
        return np.min(sess_above_8)

def get_indiv_sum_curve(d_patient, questionnaire='phq'):
    q_name = questionnaire + 'total'
    init_sess_date = d_patient[d_patient['sessionNumber']==1]['sessionDay'].iloc[0]
    final_session = get_final_session(d_patient)
    xs = []
    ys = []
    for session_id in np.arange(1, final_session+1):
        d_sess = d_patient[d_patient['sessionNumber']==session_id]
        if d_sess.shape[0] == 0:
            continue
        sum_score = d_sess[q_name].iloc[0]
        if np.isnan(sum_score):
            continue
        sess_date = d_sess['sessionDay'].iloc[0]
        day_diff = sess_date - init_sess_date
        xs.append(day_diff)
        ys.append(sum_score)
    return xs, ys

def get_indiv_item_curve(d_patient, item_idx, questionnaire='phq'):
    item_name = questionnaire + 'item' + str(item_idx)
    init_sess_date = d_patient[d_patient['sessionNumber']==1]['sessionDay'].iloc[0]
    final_session = get_final_session(d_patient)
    xs = []
    ys = []
    for session_id in np.arange(1, final_session+1):
        d_sess = d_patient[d_patient['sessionNumber']==session_id]
        if d_sess.shape[0] == 0:
            continue
        item_score = d_sess[item_name].iloc[0]
        if np.isnan(item_score):
            continue
        sess_date = d_sess['sessionDay'].iloc[0]
        day_diff = sess_date - init_sess_date
        xs.append(day_diff)
        ys.append(item_score)
    return xs, ys
   
    
    
# ---------------------------
## Plotting Helper Functions
# ---------------------------
def plot_item_lin_fit(xs, ys, slope, intercept, title='linear fit'):
    plt.figure()
    ax = plt.subplot(111)
    plt.scatter(xs, ys, marker='o', color='k', alpha=0.8, linestyle='None')
    xsfit = np.linspace(xs[0], xs[-1], 100)
    ysfit = [slope * x + intercept for x in xsfit]
    plt.plot(xsfit, ysfit, marker='None', color='k')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Item score', fontsize=12)
    plt.title(title)
    plt.ylim([-0.5, 3.5])
    plt.yticks([0, 1, 2, 3])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

        
def plot_item_exp_fit(xs, ys, a, b, c, scale='phq', title='3-param exponential fit'):
    plt.figure()
    ax = plt.subplot(111)
    plt.scatter(xs, ys, marker='o', color='k', alpha=0.8, linestyle='None')
    xsfit = np.linspace(xs[0], xs[-1], 100)
    ysfit = [exponential(x, a, b, c) for x in xsfit]
    plt.plot(xsfit, ysfit, marker='None', color='k')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Item score', fontsize=12)
    plt.title(title)
    if scale == 'pcl':
        plt.ylim([-0.5, 4.5])
        plt.yticks([0, 1, 2, 3, 4])
    else:
        plt.ylim([-0.5, 3.5])
        plt.yticks([0, 1, 2, 3])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
def plot_both_lin_exp(xs, ys, slope, intercept, a, b, c, title='Title', ylim=(-0.5, 3.5), yticks=(0,1,2,3)):
    plt.figure()
    ax = plt.subplot(111)
    plt.scatter(xs, ys, marker='o', color='k', alpha=0.8, linestyle='None')
    # Plot linear fit
    xsfit = np.linspace(xs[0], xs[-1], 100)
    ysfit = [slope * x + intercept for x in xsfit]
    plt.plot(xsfit, ysfit, marker='None', color='g', label='linear')
    # Plot exponential fit
    xsfit = np.linspace(xs[0], xs[-1], 100)
    ysfit = [exponential(x, a, b, c) for x in xsfit]
    plt.plot(xsfit, ysfit, marker='None', color='b', label='exponential')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.title(title)
    plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
def plot_both_lin_exp_pcl(xs, ys, slope, intercept, a, b, c, title='Title', ylim=(-0.5, 4.5), yticks=(0,1,2,3,4)):
    plt.figure()
    ax = plt.subplot(111)
    plt.scatter(xs, ys, marker='o', color='k', alpha=0.8, linestyle='None')
    # Plot linear fit
    xsfit = np.linspace(xs[0], xs[-1], 100)
    ysfit = [slope * x + intercept for x in xsfit]
    plt.plot(xsfit, ysfit, marker='None', color='g', label='lin')
    # Plot exponential fit
    xsfit = np.linspace(xs[0], xs[-1], 100)
    ysfit = [exponential(x, a, b, c) for x in xsfit]
    plt.plot(xsfit, ysfit, marker='None', color='b', label='exp')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Item score', fontsize=12)
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.legend()    