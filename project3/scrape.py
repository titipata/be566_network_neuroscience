import pickle
from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import defaultdict


def parse_sfn_abstract(url):
    """
    Parsing SfN page
    """
    response = urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    abstract_dict = defaultdict(list)
    tds = table.find_all('td')[2].find_all('td')
    for td in tds:
        try:
            if td.attrs.get('class')[0] == 'ViewAbstractDataLabel' and td.text.strip() != '':
                key = td.text.strip().replace(':', '').replace("(s)", '')
            elif td.attrs.get('class')[0] == 'ViewAbstractData':
                value = td.text.strip()
                abstract_dict[key].append(value)
        except:
            pass
    return abstract_dict


if __name__ == '__main__':
    ls = []
    for i, row in df.iterrows():
        try:
            d = parse_sfn_abstract(row['abstract_id'])
            d['url'].append(row['abstract_id'])
            ls.append(d)
        except:
            pass
    pickle.dump(ls, open('sfn_list.pickle', 'wb'))
