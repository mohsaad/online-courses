"""Putting it together.

Example run:
python main.py
"""

from __future__ import print_function

from io_tools import read_data_from_file, write_data_to_file
from data_tools import title_cleanup, most_frequent_words
from data_tools import most_positive_titles, most_negative_titles
from data_tools import compute_word_positivity
from data_tools import most_postivie_words, most_negative_words


def main():
    # Load data from file.
    data = read_data_from_file("data/news_data.txt")
    # Clean up data.
    title_cleanup(data)
    # Write cleaned version to file.
    write_data_to_file("data/news_data_clean.txt", data)
    # Compute most frequent words.
    print("Most Frequent Word")
    print(most_frequent_words(data)[0])
    print("--------------------------------------------")

    # Compute most positive titles
    pos_title = most_positive_titles(data)
    print("Positive title:")
    print(pos_title[0])
    print("--------------------------------------------")
    # Compute most negative titles
    neg_title = most_negative_titles(data)
    print("Negative title:")
    print(neg_title[0])
    print("--------------------------------------------")

    # Compute word positivity avg
    word_dict, word_avg = compute_word_positivity(data)
    # Compute most positive words
    pos_word_list = most_postivie_words(word_dict, word_avg)
    # Compute most negative words
    neg_word_list = most_negative_words(word_dict, word_avg)

    out_string = "List of positive words:\n"
    for word in pos_word_list:
        out_string += "%s " % word
    print(out_string + '\n')
    print("--------------------------------------------")

    out_string = "List of negative words:\n"
    for word in neg_word_list:
        out_string += "%s " % word
    print(out_string + '\n')
    print("--------------------------------------------")


if __name__ == '__main__':
    main()
