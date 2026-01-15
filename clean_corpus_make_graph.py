import heapq

import requests
import nltk
import re
import spacy
nltk.download('punkt')
nltk.download('punkt_tab')
import networkx as nx
import pandas as pd


nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

#read corpus
def read_corpus(file_path):
    with open(file_path, 'r', encoding = 'utf=8') as f:
      txt = f.read()
    return txt

def clean_corpus(text, stop_words):
  lines = text.split('\n')

  cleaned_lines = [] # for storing the cleaned lines
  skip = False
  #decided to use regex instead for efficiency
  metadata_patterns = re.compile(r'bibliography|resources|references|appendix|further reading|all rights reserved|contents', re.IGNORECASE)
  author_patterns = re.compile(r'^\s*[A-Za-z]+ \(\d{4}-\d{4}\)|^\s*By\s+[A-Za-z\s]+')
  stp = {'chapter', '.', ',', ':',';','?','-', '--', '*', 'AI', 'HTML', 'EBOOK', 'subscribe', 'email', ')', '(', "\"", "”", "“", "–", "©", "Copyright", ']', '['}

  #iterate through lines of provided text
  for line in lines:


    if skip and re.match(r'^\s*[A-Za-z]', line): #just using regexs to check
      skip = False #if paragraph, stop skip

    if skip:
      continue

    # looks for author stuff that would be formatted *author (year - year)* or *by author*
    if author_patterns.match(line):
      continue

    if line and not skip:
      cleaned_lines.append(line.strip()) #append non-metadata + non-url lines to our cleaned lines

  cleaned =  '\n'.join(cleaned_lines)

  #remove stopwords using spacy
  stpwrds = nlp(cleaned)

  #preserve sentence structure
  filtered = []
  for sent in stpwrds.sents:
    filtered_tokens = [
        token.text for token in sent
        if not token.is_stop
        and token.pos_.lower() != 'pron' and token.pos_.lower() !='aux'
        and token.text.lower() not in stp and not token.like_num
    ] #keeps only tokens that are NOTTT stopwords
    if filtered_tokens:
      filtered.append(" ".join(filtered_tokens))
  return ' '.join(filtered)

def text_to_graph(txt):
  words = txt.split()
  Graph = nx.DiGraph()

  for i in range(len(words) - 1):
    if (Graph.has_edge(words[i],
                       words[i + 1])):  # if the edge already exists, increase weight by 1, if not its weight is j 1
      Graph[words[i]][words[i + 1]]['weight'] += 1
    else:
      Graph.add_edge(words[i], words[i + 1], weight=1)
  return Graph



def make_transition_table(graph):
  edges = list(graph.edges(data = True))
  data_frame = pd.DataFrame(edges, columns=['From', 'to', 'transition attribute'])
  data_frame['transition count'] = data_frame['transition attribute'].apply(lambda x: x.get('weight', 1))#transition counts
  data_frame = data_frame[['From', 'to', 'transition count']]#table with just from, to, and transition counts

  #edge_weights = [graph[u][v]['weight'] for u, v in Graph.edges()] #get edge weights. list comprehension
  return data_frame

from pyvis.network import Network
from IPython.display import display, HTML
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib


import community as community_louvain



def cluster_color(graph):
  partition = community_louvain.best_partition(graph.to_undirected())
  for node, group in partition.items():
    graph.nodes[node]['cluster'] = group
  return graph

def top_words(graph, n):
  #most frequently visited words
  return heapq.nlargest(n, graph.nodes(data=True), key = lambda x: x[1].get('count', 0))



def draw_top_interactive(graph, top_n):
  sub_nodes = [word for word, _ in top_words(graph, top_n)]
  digraph =graph.subgraph(sub_nodes)
  clustered_g = cluster_color(digraph)

  unique_clusters = set(nx.get_node_attributes(clustered_g, "cluster").values())
  colormap = matplotlib.colormaps.get_cmap('tab10')
  color_dict = {cluster: mcolors.to_hex(colormap(i)) for i, cluster in enumerate(unique_clusters)}

  net = Network(notebook=True, directed=True, height='800px', width='100%', cdn_resources="in_line")
  for node in clustered_g.nodes:
    cluster = clustered_g.nodes[node].get("cluster", 0)  # Get cluster or default to 0
    net.add_node(node, label=node, title=f'word: {node}', group=cluster, color=color_dict.get(cluster, 'gray'))

  for edge in clustered_g.edges(data=True):
    weight = edge[2]['weight']
    net.add_edge(edge[0], edge[1], title=f'weight: {weight}' )
  html_path = 'interactive_graph.html'
  net.show(html_path)

  display(HTML(html_path))