"""
Knowledge base with Wikipedia and Wikidata,
Extends KnowledgeBaseWikipedia
Requires a Wikipedia2vec file,
a mapping from Wikipedia to Wikidata entities,
and candidate set file from Kolitsas et al. 2018

For Wikipedia2vec file, see:
https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
"""

import os
from typing import Optional
from wikipedia2vec.dictionary import Entity
from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia


class KnowledgeBaseWikidata(KnowledgeBaseWikipedia):
    def __init__(
                self,
                wikipedia2vec_file: str = '/ex_data/enwiki_20180420_100d.pkl',
                wikidata_to_wikipedia_file: str = 'wiki_wiki_mapping.tsv',
                alias_dict_file: Optional[str] = '',
                entity_dict_file: Optional[str] = '',
            ):
        """
        Alias dict and Entity dict files can be generated with
            scripts/generate_dicts_from_pem_files.py
        :param wikipedia2vec_file: .pkl file with Wikipedia2vec
        :param wikidata_to_wikipedia_file:
            file with mapping from Wikidata to Wikipedia entities
        :param alias_dict_file: .json file with alias to entity dictionary
        :param entity_dict_file: .json file with entity to alias dictionary
        """
        super().__init__(
                wikipedia2vec_file,
                alias_dict_file,
                entity_dict_file,
            )

        if not os.path.isfile(wikidata_to_wikipedia_file):
            raise FileNotFoundError(
                f"Could not find Wikidata To Wikipedia mapping file at"
                f" '{wikidata_to_wikipedia_file}'."
            )

        self.wikidata_to_wikipedia = {}
        self.wikipedia_to_wikidata = {}
        with open(wikidata_to_wikipedia_file) as ww_file:
            for line in ww_file:
                entities = line[:-1].split('\t')
                wikidata_entity = entities[0]
                wikipedia_entity = entities[1]
                self.wikipedia_to_wikidata[wikipedia_entity] = wikidata_entity
                self.wikidata_to_wikipedia[wikidata_entity] = wikipedia_entity

    def get_candidate_set_from_mention(self, mention_text: str):
        """
        Get candidates for mention text
        :param mention_text: alias text
        :returns: a list of candidates for the mention text
        """
        candidates = super().get_candidate_set_from_mention(mention_text)
        candidates = [self.wikipedia_to_wikidata[cand] for cand in candidates
                      if cand in self.wikipedia_to_wikidata
                      and self.get_w2v_entity(
                            self.wikipedia_to_wikidata[cand]
                          ) is not None]
        return candidates

    def get_candidate_set_from_vector(self, vector):
        """
        Get most similar entities (candidates) for vector
        :param vector: search vector
        :returns: a list of 10 most similar entitites to the vector
        """
        # Get Wikidata ID of candidates
        candidates = \
            self.wikipedia2vec.most_similar_by_vector(vector, count=50)

        similar_entities = []
        for candidate in candidates:
            if isinstance(candidate[0], Entity):
                # Ignoring entities that are not in the knowledge base
                if candidate[0].title not in self.wikipedia_to_wikidata:
                    continue
                wikidata_id = \
                    self.wikipedia_to_wikidata[candidate[0].title]
                score = candidate[1]
                similar_entities += [(wikidata_id, score)]
            if len(similar_entities) >= 10:
                break

        similar_entities = \
            sorted(similar_entities, reverse=True, key=lambda x: (x[1]))
        return similar_entities

    def get_w2v_entity(self, entity):
        """
        :param entity: the entity ID as returned by alias_dict
        :returns: the Wikipedia2vec entity title after redirect.
            use this to find e.g. Wikipedia2vec vectors
        """
        if entity in self.wikidata_to_wikipedia:
            wikipedia_entity = self.wikidata_to_wikipedia[entity]
        elif entity in self.wikipedia_to_wikidata:
            wikipedia_entity = entity
        else:
            return None

        return super().get_w2v_entity(wikipedia_entity)

    def get_kb_entity(self, entity: str):
        """
        Returns the unidecoded entity ID used for this KB
        :param entity: the entity ID
        :returns: the entity ID as used in entity_dict or alias_dict
        """
        if entity in self.wikidata_to_wikipedia:
            return entity
        else:
            return None

    def in_cand_gen(self, entity: str):
        """
        :param entity: a KB entity ID, as returned by self.get_kb_entity
        :returns: true if entity is in the Candidate Generation mapping
        """
        if not self.cand_gen:
            return False
        entity = self.wikidata_to_wikipedia[self.get_kb_entity(entity)]
        return bool(entity in self.entity_dict)
