'''
Convert the file prob_yago_crosswikis_wikipedia_p_e_m.txt
    (accessed through config.ini - 'KNOWLEDGE BASE', 'Alias Mapping')
    into two .json files:
     * A file 'alias_dict' with mappings from alias to candidate sets
     * A file 'entity_dict' with mappings from entity to aliases
'''

from collections import defaultdict
from configparser import ConfigParser
import json
from unidecode import unidecode
from wikipedia2vec import Wikipedia2Vec
from os import path


config = ConfigParser()
config.read('config.ini')
p_e_m_file = config['KNOWLEDGE BASE']['Alias Mapping']
w2v_file = path.join(
        config['KNOWLEDGE BASE']['Wikipedia2vec Directory'],
        config['KNOWLEDGE BASE']['Wikipedia2Vec File'],
    )
wikipedia2vec = Wikipedia2Vec.load(w2v_file)

cands_in_w2w = 0
cands_not_in_w2w = 0
# Get the file length for progress updates
with open(p_e_m_file) as pem_file:
    n_lines = sum(1 for _ in pem_file)

alias_dict = defaultdict(set)
entity_dict = defaultdict(list)

# Iterate file
pem_file = open(p_e_m_file)
for i_line, line in enumerate(pem_file):
    values = unidecode(line.strip()).split('\t')
    # Normalize and lower-case the mention
    alias = values[0].lower()
    candidates = values[2:]
    for cand in candidates:
        # Take commas in the title into account,
        #   and replace underscore by spaces
        cand = ','.join(cand.split(',')[2:]).replace('_', ' ')
        # Get the Wikipedia2vec entity, if it exists
        entity = wikipedia2vec.get_entity(cand)
        # If entity is in Wikipedia2vec
        if entity:
            # Use the title of the Wikpedia2vec entity (after redirect)
            cand_title = unidecode(entity.title)
            # Add candidate to alias-candidate dict
            alias_dict[alias].add(cand_title)
            # And add alias to entity-alias dict
            entity_dict[cand_title] += [alias]
            cands_in_w2w += 1
        else:
            cands_not_in_w2w += 1
    if (i_line + 1) % 100000 == 0:
        print("Read {:>8} / {} lines of Alias Mapping".format(
                i_line + 1,
                n_lines,
            ), end='\r')

pem_file.close()

for mention in alias_dict.keys():
    alias_dict[mention] = list(alias_dict[mention])

alias_dict_file = open(config['KNOWLEDGE BASE']['Alias Dict'], 'w')
entity_dict_file = open(config['KNOWLEDGE BASE']['Entity Dict'], 'w')
json.dump(alias_dict, alias_dict_file)
json.dump(entity_dict, entity_dict_file)
alias_dict_file.close()
entity_dict_file.close()

print(f"Wrote Alias Dict to {alias_dict_file.name}")
print(f"Wrote Entity Dict to {entity_dict_file.name}")

print(f"Read {i_line + 1:>8} / {n_lines} lines of Alias Mapping")
print(f"Got {cands_in_w2w} candidates, and missed {cands_not_in_w2w}")
print()
