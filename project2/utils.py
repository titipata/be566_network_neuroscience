import re
import bct
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter


def parse_sub_topic_pair(subtopic):
    """
    Parse topic and return topic and topic name
    """
    if pd.isnull(subtopic):
        subtopic = ''
    subtopic = re.sub('A.02 c.', 'A.02.c.', subtopic)
    subtopic = re.sub('A.02 b.', 'A.02.b.', subtopic)
    subtopic_title = re.search('\w+\.\w+\.\w+', subtopic)
    if subtopic_title:
        topic_tree = subtopic_title.group()
        topic_name = re.sub(topic_tree, '', subtopic).replace('.', '').strip()
        return [subtopic_title.group(), topic_name]
    else:
        return ['', '']


def hamming_similarity(st1, st2):
    """
    Hamming similarity between two SfN subtopic e.g. A.01.r, A.01.e
    """
    theme1, topic1, subtopic1 = st1.split('.')
    theme2, topic2, subtopic2 = st2.split('.')
    if theme1 == theme2 and topic1 == topic2 and subtopic1 == subtopic2:
        return 3
    elif theme1 == theme2 and topic1 == topic2 and subtopic1 != subtopic2:
        return 2
    elif theme1 == theme2 and topic1 != topic2 and subtopic1 != subtopic2:
        return 1
    else:
        return 0


def hamming_similarity_poster(p1, p2):
    """
    Hamming similarity between 2 posters each have tuple of subtopics
    such as ('A.01.r', 'A.01.e')
    """
    sim = hamming_similarity(p1[0], p2[0]) + \
        hamming_similarity(p1[0], p2[1]) + \
        hamming_similarity(p1[1], p2[0]) + \
        hamming_similarity(p1[1], p2[1])
    return sim


def gini(list_of_values):
    """
    Gini inequality of list of values
    """
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area


if __name__ == '__main__':
    theme_df = pd.ExcelFile('SfN2017_Control_Theme_Data.xlsx').parse('Sheet1')
    theme_df = theme_df[~theme_df.theme1.isnull()]
    sfn_df = pd.read_csv('SfN2017_RecommendationEngine.csv', encoding = "ISO-8859-1")
    abstract_df = pd.ExcelFile('SfN2017_ControlAbstract.xlsx').parse('Sheet1')

    sfn_df.columns = [
        'control_number', 'presentation_id', 'presentation_number',
        'start_time', 'title', 'abstract', 'author', 'location'
    ]
    theme_df.columns = [
        'control_number', 'theme1', 'theme1enum', 'topic1',  'topic1enum',
        'subtopic1', 'subtopic1enum',  'theme2', 'theme2enum', 'topic2',
        'topic2enum', 'subtopic2', 'subtopic2enum', 'keyword', 'ric_keyword',
        'preference', 'prefernece_enum'
    ]
    abstract_df.columns = ['control_number', 'abstract']

    abstract_control_numbers = set(sfn_df['Abstract.ControlNumber'].unique())
    theme_control_numbers = set(theme_df.ControlNumber.unique())

    subtopic_split_1 = theme_df.subtopic1.map(lambda x: parse_sub_topic_pair(x)[0])
    subtopic_split_2 = theme_df.subtopic2.map(lambda x: parse_sub_topic_pair(x)[0])
