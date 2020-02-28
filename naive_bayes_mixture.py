# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter


def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda, unigram_smoothing_parameter,
                      bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """
    cnt_pos = Counter()
    cnt_neg = Counter()
    bi_cnt_pos = Counter()
    bi_cnt_neg = Counter()
    log_likelihood_pos = dict()
    log_likelihood_neg = dict()
    bi_log_likelihood_pos = dict()
    bi_log_likelihood_neg = dict()
    for ind, file in enumerate(train_set):
        if train_labels[ind] == 1:
            for word in file:
                cnt_pos[word] += 1
                cnt_neg[word] += 0
            if len(file) >= 2:
                for i in range(len(file) - 1):
                    bigram = file[i] + ' ' + file[i + 1]
                    bi_cnt_pos[bigram] += 1
                    bi_cnt_neg[bigram] += 0
        else:
            for word in file:
                cnt_neg[word] += 1
                cnt_pos[word] += 0
            if len(file) >= 2:
                for i in range(len(file) - 1):
                    bigram = file[i] + ' ' + file[i + 1]
                    bi_cnt_neg[bigram] += 1
                    bi_cnt_pos[bigram] += 0
    for file in dev_set:
        for word in file:
            cnt_pos[word] += 0
            cnt_neg[word] += 0
        if len(file) >= 2:
            for i in range(len(file) - 1):
                bigram = file[i] + ' ' + file[i + 1]
                bi_cnt_pos[bigram] += 0
                bi_cnt_neg[bigram] += 0
    unique_pos = len(list(cnt_pos))
    unique_neg = len(list(cnt_neg))
    bi_unique_pos = len(list(bi_cnt_pos))
    bi_unique_neg = len(list(bi_cnt_neg))
    cnt_pos_total = sum(cnt_pos.values())
    cnt_neg_total = sum(cnt_neg.values())
    bi_cnt_pos_total = sum(cnt_pos.values())
    bi_cnt_neg_total = sum(cnt_neg.values())
    for word in list(cnt_pos):
        log_likelihood_pos[word] = math.log(float(cnt_pos[word] + unigram_smoothing_parameter) / (
                unigram_smoothing_parameter * unique_pos + cnt_pos_total))
        log_likelihood_neg[word] = math.log(float(cnt_neg[word] + unigram_smoothing_parameter) / (
                unigram_smoothing_parameter * unique_neg + cnt_neg_total))
    for bigram in list(bi_cnt_pos):
        bi_log_likelihood_pos[bigram] = math.log(float(bi_cnt_pos[bigram] + bigram_smoothing_parameter) / (
                bigram_smoothing_parameter * bi_unique_pos + bi_cnt_pos_total))
        bi_log_likelihood_neg[bigram] = math.log(float(bi_cnt_neg[bigram] + bigram_smoothing_parameter) / (
                bigram_smoothing_parameter * bi_unique_neg + bi_cnt_neg_total))

    neg_prior = 1 - pos_prior
    dev_labels = []
    for file in dev_set:
        log_pos_posterior = math.log(pos_prior)
        log_neg_posterior = math.log(neg_prior)
        bi_log_pos_posterior = math.log(pos_prior)
        bi_log_neg_posterior = math.log(neg_prior)
        for word in file:
            log_pos_posterior += log_likelihood_pos[word]
            log_neg_posterior += log_likelihood_neg[word]
        if len(file) >= 2:
            for i in range(len(file) - 1):
                bigram = file[i] + ' ' + file[i + 1]
                bi_log_pos_posterior += bi_log_likelihood_pos[bigram]
                bi_log_neg_posterior += bi_log_likelihood_neg[bigram]
        else:
            bi_log_pos_posterior = 0
            bi_log_neg_posterior = 0
        if (1 - bigram_lambda) * log_pos_posterior + bigram_lambda * bi_log_pos_posterior > (
                1 - bigram_lambda) * log_neg_posterior + bigram_lambda * bi_log_neg_posterior:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels
