"""
Knowledge Base with
  Wikipedia2vec and
  AIDA Candidate Sets from Kolitsas et al. 2018

Requires:
 * a Wikipedia2vec embedding file
    (see: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)
 For Candidate Generation:
 * alias_dict.json file
 * entity_dict.json file
 These are generated with:
python -m scripts.generate_dicts_from_pem_file

"""

import numpy
from typing import Optional
from collections import defaultdict
from unidecode import unidecode
from wikipedia2vec import Wikipedia2Vec
from wikipedia2vec.dictionary import Entity
from json import loads


class KnowledgeBaseWikipedia():
    def __init__(
                self,
                wikipedia2vec_file: str = '/ex_data/enwiki_20180420_100d.pkl',
                alias_dict_file: Optional[str] = '',
                entity_dict_file: Optional[str] = '',
            ):
        """
        Alias dict and Entity dict files can be generated with
            scripts/generate_dicts_from_pem_files.py
        :param wikipedia2vec_file: .pkl file with Wikipedia2vec
        :param alias_dict_file: .json file with alias to entity dictionary
        :param entity_dict_file: .json file with entity to alias dictionary
        """
        self.wikipedia2vec = Wikipedia2Vec.load(wikipedia2vec_file)
        self.resolve_redirect = True
        self.dim_size = self.wikipedia2vec.syn0.shape[1]

        self.cand_gen = False
        self.alias_dict = defaultdict(set)
        self.entity_dict = defaultdict(list)

        if alias_dict_file and entity_dict_file:
            alias_file = open(alias_dict_file)
            self.alias_dict = loads(alias_file.read())
            alias_file.close()
            entity_file = open(entity_dict_file)
            self.entity_dict = loads(entity_file.read())
            entity_file.close()
            self.alias_dict = defaultdict(set, self.alias_dict)
            self.entity_dict = defaultdict(list, self.entity_dict)
            self.cand_gen = True

    def find_similar(
                self,
                input_vector,
                mention_text: str = '',
                use_fallback_for_empty_cs: bool = True
            ):
        """
        Return similar vectors to input vector,
        and their score.

        :param input_vector: the embedded entity vector used as search key
            in Wikipedia2vec (must be convertible as numpy array)
        :param mention_text: if provided, triggers candidate generation
        :param use_fallback_for_empty_cs: if True,
            and using candidate generation, using fallback to brute-force
            search over all Wikipedia2vec vectors. This is slower,
            and not recommended during training.
        :returns: the most similar entity
        """
        # Cast to numpy array
        input_vector = numpy.array(input_vector).flatten()
        # Use candidate sets, if mentions are provided
        if mention_text != '':
            # Get Wikidata ID of candidates
            return self.get_candidate_set(
                    input_vector,
                    mention_text,
                    use_fallback_for_empty_cs
                )
        # No candidate sets
        else:
            # Getting most similar vectors
            return self.get_candidate_set_from_vector(input_vector)

    def get_candidate_set(
                self,
                vector,
                mention_text,
                use_fallback_for_empty_cs,
            ):
        """
        Get candidate set for a given mention text,
            and similarity to vector

        :param vector: the query vector used for similarity
        :param mention_text: alias text
        :param use_fallback_for_empty_cs: if True, using brute-force
            search over all Wikipedia2vec vectors when no entities for alias.
            Not recommended during training.
        :returns: a list of candidates for the mention text sorted by
            similarity to vector
        """
        # Get Wikidata ID of candidates
        candidates = self.get_candidate_set_from_mention(mention_text)
        if len(candidates) > 0:
            cands = self.candidate_similarity(vector, candidates)
            return cands
        # If there were no candidates, use brute-force similarity search
        elif use_fallback_for_empty_cs:
            return self.get_candidate_set_from_vector(vector)
        else:
            return []

    def get_candidate_set_from_mention(self, mention_text: str):
        """
        Get candidates for mention text
        :param mention_text: alias text
        :returns: a list of candidates for the mention text
        """
        mention_normalized = unidecode(mention_text).lower()
        candidates = self.alias_dict[mention_normalized]
        candidates = [cand for cand in candidates
                      if self.get_w2v_entity(cand) is not None]
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
                score = candidate[1]
                similar_entities += [(unidecode(candidate[0].title), score)]
            if len(similar_entities) >= 10:
                break

        similar_entities = \
            sorted(similar_entities, reverse=True, key=lambda x: (x[1]))
        return similar_entities

    def candidate_similarity(self, input_vector, candidate_entities):
        """
        Finds the similarity between candidate vectors and input vector
        :param input_vector: the embedded entity vector used as search key
            in Wikipedia2vec (must be convertible as numpy array)
        :param candidate_entities: the candidate list
        :returns: a sorted list of tuples with candidates and similarity score
        """
        # Get the entity vectors of the candidates
        candidate_vectors = []
        for candidate in candidate_entities:
            candidate_vector = self.get_entity_vector(candidate)
            candidate_vectors += [candidate_vector]
        candidate_vectors = numpy.array(candidate_vectors)

        # Vector-matrix multiplication for similarity score
        # Through broadcasting, this gives the dot-product
        #  of the vector with each candidate's vector
        score = numpy.dot(candidate_vectors, input_vector).flatten()
        # Normalize the input vector by dividing by 2-norm
        vec_norm = numpy.linalg.norm(input_vector, ord=2)
        # 2-norm of each entity vector
        cand_norm = numpy.linalg.norm(candidate_vectors, ord=2, axis=1)
        score = score / cand_norm / vec_norm
        similar_entities = list(zip(candidate_entities, score))
        # Sort candidates by score
        similar_entities = \
            sorted(similar_entities, reverse=True, key=lambda x: (x[1]))
        return similar_entities

    def get_entity_vector(self, entity: str):
        """
        Get the wikipedia2vec vector of an entity
        :param entity: the entity ID
        :returns: the embedding vector of the entity, or None
            if the entity is not in the knowledge base.
        """
        w2v_ent = self.get_w2v_entity(entity)
        if w2v_ent is None:
            return None
        vector = self.wikipedia2vec.get_entity_vector(
                w2v_ent, resolve_redirect=self.resolve_redirect
            )
        return vector

    def get_w2v_entity(self, entity):
        """
        :param entity: the entity ID as returned by alias_dict
        :returns: the Wikipedia2vec entity title after redirect.
            use this to find e.g. Wikipedia2vec vectors
        """
        entity = entity.replace('_', ' ')
        w2v_ent = self.wikipedia2vec.get_entity(entity)
        if w2v_ent:
            return w2v_ent.title
        else:
            return None

    def get_kb_entity(self, entity: str):
        """
        Returns the unidecoded entity ID used for this KB
        :param entity: the entity ID
        :returns: the entity ID as used in entity_dict or alias_dict
        """
        w2v_ent = self.get_w2v_entity(entity)
        if w2v_ent:
            decoded = unidecode(w2v_ent)
            # This is a hotfix (which seems to work)
            if self.in_kb(decoded):
                return decoded
            else:
                return w2v_ent
        else:
            return None

    def in_kb(self, entity: str):
        """
        :param entity: a KB entity ID, as returned by self.get_kb_entity
        :returns: True if entity is in Wikipedia2vec
        """
        return bool(self.get_w2v_entity(entity))

    def in_cand_gen(self, entity: str):
        """
        :param entity: a KB entity ID, as returned by self.get_kb_entity
        :returns: true if entity is in the Candidate Generation mapping
        """
        if not self.cand_gen:
            return False
        return bool(self.get_kb_entity(entity) in self.entity_dict)

    def n_cg_entities(self):
        """
        :returns: number of unique entities in the Candidate Generation
        """
        return len(self.entity_dict)

    def n_cg_aliases(self):
        """
        :returns: number of aliases in the Candidate Generation
        """
        return len(self.alias_dict)

    def n_entities(self):
        """
        :returns: number of entities in current Wikipedia2vec
        """
        return sum(1 for _ in self.wikipedia2vec.dictionary.entities())
