import re
import dedupe
from unidecode import unidecode
from bs4 import BeautifulSoup
import pandas as pd
from itertools import chain, combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def parse_text_with_sup(text):
    """
    Parse given affiliation text to list of author and affiliation
    """
    authors = text.split('; <br/>')[0]
    affiliation = text.split('; <br/>')[1]
    affiliations = [a for a in affiliation.split('<sup>') if a.strip() is not '']
    affil_dict = {}
    for a in affiliations:
        sup, affil =  a.split('</sup>')
        affil_dict[sup] = affil

    author_affil_list = []
    for a in authors.split(', '):
        soup = BeautifulSoup(a, 'html.parser')
        try:
            sup = soup.find('sup').text
        except:
            sup = ''
        affil_author = affil_dict.get(sup, '')
        author = soup.text
        author = re.sub(sup, '', author).replace(',', '')
        author = re.sub('[\d+]', '', author)
        author_affil_list.append((author, affil_author))
    return author_affil_list


def parse_sfn_author(text):
    try:
        authors, affliation = text.split('; <br/>')
        replace_dict = {', III': ' III', ', IV': ' IV', ', II': ' II'}
        for (r1, r2) in replace_dict.items():
            authors = authors.replace(r1, r2)
        authors = BeautifulSoup(authors, 'html.parser').text
        if '<sup>' in text:
            return parse_text_with_sup(text)
        elif ',' not in authors:
            authors = authors
            affiliation = BeautifulSoup(affliation, 'html.parser').text
            return [(authors, affiliation)]
        elif ',' in authors:
            authors = authors.split(', ')
            affiliation = BeautifulSoup(affliation, 'html.parser').text
            return [(a, affiliation) for a in authors]
        else:
            return []
    except:
        return []


def apply_dedupe(authors):
    """
    Give list of authors and publication id as
        [(presentation_id, name, affiliation), ...]
    Tag using dedupe library and return
    """
    fields = [{'field': 'Name', 'type': 'String', 'has missing': True},
              {'field': 'Affiliation', 'type': 'String', 'has missing': True}]
    deduper = dedupe.Dedupe(fields)

    # prepare dedupe dictionary
    author_dict = {}
    for i, author in enumerate(authors):
        if author[1] != '' and author[2] != '':
            author_name = unidecode(author[1] or '').replace('-.', ' ').replace('.', '').replace('-', ' ').upper()
            affiliation_name = unidecode(author[2] or '').replace(',', '').replace('.', '').lower()
            author_dict[i] = {'PaperId': author[0],
                              'Name': author_name,
                              'Affiliation': affiliation_name}
    deduper.sample(author_dict, 1000)
    dedupe.consoleLabel(deduper)
    deduper.train()
    threshold = deduper.threshold(author_dict, recall_weight=1.0)
    clustered_dupes = deduper.match(author_dict, threshold)
    clustered_dupes_pair = [pair for pair, _ in clustered_dupes] # pairs

    author_ids = []
    for i, pair in enumerate(clustered_dupes_pair):
        author_ids.extend([(i, e) for e in pair])
    authors_unique = list(set(range(len(authors))) - set([author_id for _, author_id in author_ids]))

    aid = author_ids[-1][0]
    for author in authors_unique:
        aid = aid + 1
        author_ids.append((aid, author))
    author_id_df = pd.DataFrame(author_ids,
                                columns=['author_id', 'author_number'])
    return author_id_df


def create_collaboration_graph(author_df):
    """
    Create graph where each nodes is unique author and each edges is collaboration
    """
    nodes_df = author_df[['affiliation', 'author_name', 'author_id']].drop_duplicates(subset='author_id')
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row['author_id'], attr_dict={'affiliation': row['affiliation'],
                                                'author_name': row['author_name']})
    # get papers with more than 2 authors
    n_papers_df = author_df[['paper_id']].groupby('paper_id').size().reset_index()
    n_papers_df.columns = ['paper_id', 'n_authors']
    n_papers_df = n_papers_df[n_papers_df.n_authors > 1]

    # add edges
    edges = []
    for paper_id in n_papers_df.paper_id:
        authod_ids = list(author_df[author_df.paper_id == paper_id].author_id)
        edges.extend(list(combinations(authod_ids, 2)))
    for n1, n2 in edges:
        G.add_edge(n1, n2)
    return G


def hist_publication(df):
    """
    Using Altair to plot histogram of chart
    """
    chart = Chart(df).mark_bar(barSize=20).encode(
        color=Color(value='#fbb33a'),
        x=X('n_authors:Q', bin=Bin(maxbins=30), title='Number of authors in a poster'),
        y=Y('count(*):Q', title='Number of posters'),
    ).configure_facet_cell(
        strokeWidth=0.0,
    )
    return chart


if __name__ == '__main__':
    df = pd.read_csv('SfN2017_RecommendationEngine.csv')
    df.columns = ['presentation_id', 'presentation_number', 'start_time', 'title', 'abstract', 'author', 'location']
    df.fillna('', inplace=True)
    df['parsed_authors'] = df.author.map(parse_sfn_author)

    authors = list()
    for i, row in df.iterrows():
        for a in row['parsed_authors']:
            if len(row['parsed_authors']) > 0 and a[1] is not '':
                authors.append((row['presentation_id'], str(a[0].lower().replace('*', '')), a[1].lower()))

    # applying deduplication algorithm
    author_id_df = apply_dedupe(authors)

    author_number_list = []
    for i, (paper_id, author_name, affiliation) in enumerate(authors):
        author_number_list.append({'author_number': i,
                                   'paper_id': paper_id,
                                   'author_name': author_name,
                                   'affiliation': affiliation})
    author_number_df = pd.DataFrame(author_number_list)
    author_df = author_number_df.merge(author_id_df, how='left') # final author number
    author_df.to_csv('author_id.csv', index=False)

    G = create_collaboration_graph(author_df)
    bet_cen = nx.betweenness_centrality(G) # betweenness centrality
    deg_cen = nx.degree_centrality(G) # degree centrality
    ben_cen_df = pd.DataFrame(list(bet_cen.items()), columns=['author_id', 'betweenness_centrality'])
    deg_cen_df = pd.DataFrame(list(deg_cen.items()), columns=['author_id', 'degree_centrality'])
    centrality_df = ben_cen_df.merge(deg_cen_df)
    centrality_df.to_csv('centrality.csv', index=False)
