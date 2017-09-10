"""Processing data tools for mp0.
"""
import re
import numpy as np
from operator import itemgetter
from string import digits, punctuation


def title_cleanup(data):
    """Remove all characters except a-z, A-Z and spaces from the title,
       then convert all characters to lower case.

    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    """
    for key in data:
        data[key][0] = data[key][0].translate(None, punctuation).translate(None, digits).lower()

def most_frequent_words(data):
    """Find the more frequeny words (including all ties), returned in a list.

    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        max_words(list): List of strings containing the most frequent words.
    """
    max_words = []
    max_words_dict = {}
    for key in data:
        data_str = data[key][0].split(" ")
        for word in data_str:
            if word in max_words_dict:
                max_words_dict[word] += 1
            else:
                max_words_dict[word] = 1

    values = max_words_dict.values()
    num_words ={}

    for key in max_words_dict:
        if(max_words_dict[key] not in num_words):
            num_words[max_words_dict[key]] = [key]
        else:
            num_words[max_words_dict[key]].append(key)

    max_word_nums = []
    for key in num_words:
        max_word_nums.append((key, num_words[key]))

    max_word_nums = sorted(max_word_nums, key = itemgetter(0), reverse = True)

    for i in range(0, len(max_word_nums)):
        max_words.append(max_word_nums[i][1])

    return max_words


def most_positive_titles(data):
    """Computes the most positive titles.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        titles(list): List of strings containing the most positive titles,
                      include all ties.
    """
    titles = []
    title_dict = {}
    for key in data:
        if data[key][1] not in title_dict:
            title_dict[data[key][1]] = [data[key][0]]
        else:
            title_dict[data[key][1]].append(data[key][0])

    total_titles = []
    for key in title_dict:
        total_titles.append((key, title_dict[key]))

    max_score_titles = sorted(total_titles, key=itemgetter(0), reverse = True)
    for i in range(0, len(max_score_titles)):
        titles.append(max_score_titles[i][1])
    return titles


def most_negative_titles(data):
    """Computes the most negative titles.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
     Returns:
        titles(list): List of strings containing the most negative titles,
                      include all ties.
    """
    titles = []
    title_dict = {}
    for key in data:
        if data[key][1] not in title_dict:
            title_dict[data[key][1]] = [data[key][0]]
        else:
            title_dict[data[key][1]].append(data[key][0])

    total_titles = []
    for key in title_dict:
        total_titles.append((key, title_dict[key]))

    max_score_titles = sorted(total_titles, key=itemgetter(0))
    for i in range(0, len(max_score_titles)):
        titles.append(max_score_titles[i][1])
    return titles


def compute_word_positivity(data):
    """Computes average word positivity.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        word_dict(dict): Key: word(str), value: word_index(int)
        word_avg(numpy.ndarray): numpy array where element
                                 #word_dict[word] is the
                                 average word positivity for word.
    """
    word_dict = {}
    avg_word_dict = {}

    # get dict of all words
    for key in data:
        data_str = data[key][0].split(" ")
        for word in data_str:
            if word in avg_word_dict:
                avg_word_dict[word][0] += data[key][1]
                avg_word_dict[word][1] += 1
            else:
                avg_word_dict[word] = [data[key][1],1]

    word_avg = []
    count = 0

    for key in avg_word_dict:
        word_avg.append(avg_word_dict[key][0] / avg_word_dict[key][1])
        word_dict[key] = count
        count += 1

    return word_dict, np.array(word_avg)


def most_postivie_words(word_dict, word_avg):
    """Computes the most positive words.
    Args:
        word_dict(dict): output from compute_word_positivity.
        word_avg(numpy.ndarray): output from compute_word_positivity.
    Returns:
        words(list):
    """
    words = []
    pos_words = []
    for key in word_dict:
        pos_words.append((key, word_avg[word_dict[key]]))

    pos_words = sorted(pos_words, key = itemgetter(1), reverse = True)

    count = 0
    for i in range(0, len(pos_words)):
        if i == 0:
            words.append([pos_words[i][0]])
        else:
            if(pos_words[i][1] == pos_words[i-1][1]):
                words[count].append(pos_words[i][0])
            else:
                count += 1
                words.append([])
                words[count].append(pos_words[i][0])

    return words


def most_negative_words(word_dict, word_avg):
    """Computes the most negative words.
    Args:
        word_dict(dict): output from compute_word_positivity.
        word_avg(numpy.ndarray): output from compute_word_positivity.
    Returns:
        words(list):
    """
    words = []
    pos_words = []
    for key in word_dict:
        pos_words.append((key, word_avg[word_dict[key]]))

    pos_words = sorted(pos_words, key = itemgetter(1), reverse = False)

    count = 0
    for i in range(0, len(pos_words)):
        if i == 0:
            words.append([pos_words[i][0]])
        else:
            if(pos_words[i][1] == pos_words[i-1][1]):
                words[count].append(pos_words[i][0])
            else:
                count += 1
                words.append([])
                words[count].append(pos_words[i][0])

    return words
