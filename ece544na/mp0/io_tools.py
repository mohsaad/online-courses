"""IO tools for mp0.
"""


def read_data_from_file(filename):
    """
    Read txt data from file.
    Each row is in the format article_id\ttitle\tpositivity_score\n.
    Store this information in a python dictionary. Key: article_id(int),
    value: [title(str), score(float)].

    Args:
        filename(string): Location of the file to load.
    Returns:
        out_dict(dict): data loaded from file.
    """
    out_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            line_data = line.split('\n')[0].split('\t')
            article_id = int(line_data[0])
            out_dict[article_id] = [line_data[1], float(line_data[2])]

    return out_dict


def write_data_to_file(filename, data):
    """
    Writes data to file in the format article_id\ttitle\tpositivity_score\n.

    Args:
        filename(string): Location of the file to save.
        data(dict): data for writting to file.
    """
    out_f = open(filename, 'w')
    for key in data:
        out_f.write("{0}\t{1}\t{2}\n".format(key, data[key][0], data[key][1]))
    out_f.close()
