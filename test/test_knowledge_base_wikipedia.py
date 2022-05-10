import unittest
from configparser import ConfigParser
from numpy.testing import assert_almost_equal
from numpy import array as nparray
from src.toolbox import get_knowledge_base


class TestKnowledgeBaseWikipedia(unittest.TestCase):

    def setUp(self):
        self.config = ConfigParser()
        self.config.read('config.ini')

    def tearDown(self):
        self.config = None

    def test_init_with_cg(self):
        """Test initializing KB with Alias and Entity dict files"""
        subject = get_knowledge_base(self.config, wiki='pedia', with_cg=True)
        assert subject.cand_gen is True
        test_find_similar_cg(subject)

    def test_find_similar_no_cg(self):
        """Test finding similar entities by vector (no CG)"""
        subject = get_knowledge_base(self.config, wiki='pedia', with_cg=False)
        vector = subject.wikipedia2vec.get_entity_vector(
                'Scarlett Johansson'
            )

        expected_qids = [
                'Scarlett Johansson',
                'Hilary Swank',
                'Eva Green',
                'Kate Winslet',
                'Keira Knightley'
            ]
        expected_scores = [1.0, 0.7215545, 0.71551895, 0.71038455, 0.7100761]

        result = subject.find_similar(vector)[:5]

        result_qids = [r[0] for r in result]
        result_scores = [r[1] for r in result]

        self.assertEqual(result_qids, expected_qids)
        assert_almost_equal(result_scores, expected_scores, decimal=7)

    def test_get_w2v_entity(self):
        """Test getting w2v entity"""
        subject = get_knowledge_base(self.config, wiki='pedia', with_cg=False)
        entity = 'The_Adventures_of_Tintin'
        expected = 'The Adventures of Tintin'
        res = subject.get_w2v_entity(entity)
        self.assertEqual(expected, res)

    def test_get_kb_entity(self):
        """Test getting w2v entity"""
        subject = get_knowledge_base(self.config, wiki='pedia', with_cg=False)
        entity = u'Ålesund'
        expected = 'Ålesund'
        res = subject.get_kb_entity(entity)
        self.assertEqual(expected, res)

    def test_in_kb(self):
        """Test getting w2v entity"""
        subject = get_knowledge_base(self.config, wiki='pedia', with_cg=False)
        entity = 'Feist (singer)'
        res = subject.in_kb(entity)
        self.assertTrue(res)

    def test_in_cand_gen(self):
        """Test getting w2v entity"""
        subject = get_knowledge_base(self.config, wiki='pedia', with_cg=True)
        entity = 'Feist (singer)'
        res = subject.in_cand_gen(entity)
        self.assertTrue(res)

    def test_get_entity_vector(self):
        """Test getting Wikipedai2vec entity vector with KB function"""
        subject = get_knowledge_base(self.config, wiki='pedia', with_cg=False)
        
        entity_in_kb = 'Scarlett Johansson'
        expected_in_kb = nparray([
                -0.8400287, 0.5288666, 0.04453477, -0.54837024, -0.9410379,
                0.11284031, -0.17867407, 1.3697083, -0.03290863, -0.582104,
                -0.17381532, 0.24788703, 1.167423, -0.9521753, -0.48488948,
                -0.42186424, 0.8235442, 0.7793223, 0.77660275, 0.25757137,
                1.3587517, -1.4913211, -0.72050506, 0.599874, -0.6053576,
                0.14196505, -1.0757631, 0.3259684, 0.3301182, 2.5082862,
                0.649137, 0.08131427, 0.7075302, 0.3335169, 0.61395925,
                -0.14824118, -0.97298646, -0.45351806, -0.9709426, -0.27593935,
                -0.01545926, -0.08553899, -1.3710245, -0.13411002, -0.5522892,
                -0.7656844, -0.56291217, 0.46116182, -0.97181046, -0.03204511,
                1.1024369, 1.2622716, -0.28116563, 0.3250025, 0.30856025,
                0.03439774, -1.0911576, -0.49705723, -0.18033183, -0.19287291,
                1.396642, -0.31962425, 0.26252577, 0.43745142, 0.9549161,
                -0.8271081, -1.2802972, -0.02027902, -0.31275767, -0.43658108,
                0.4247029, -0.28418005, -0.16575934, -0.73438936, -1.3567144,
                -0.4372178, 1.2283757, -0.0503046, -0.79971427, 0.05765368,
                0.04880399, 0.02459021, 1.1055263, 0.9883592, -0.52239317,
                -0.32314008, 1.6714385, -0.41210583, 1.22322, 0.32987827,
                -0.24956056, 1.2802523, 0.7738934, -0.19011438, -0.8993519,
                0.03893373, 0.39385346, 0.502828, 0.02950748, -0.02772579
            ])
        result_in_kb = subject.get_entity_vector(entity_in_kb)

        entity_not_in_kb = 'Gilles Lussier'
        assert not subject.in_kb(entity_not_in_kb), \
            f"The test case {entity_not_in_kb} was in fact in the KB!"
        expected_not_in_kb = None
        result_not_in_kb = subject.get_entity_vector(entity_not_in_kb)

        assert_almost_equal(expected_in_kb, result_in_kb, decimal=7)
        self.assertEqual(expected_not_in_kb, result_not_in_kb)


def test_find_similar_cg(subject):
    """Test finding similar entities by mention text (using CG)"""
    vector = subject.wikipedia2vec.get_entity_vector(
            'Scarlett Johansson'
        )
    mention_text = 'Scarlett Johansson'
    expected_ids = [
            'Scarlett Johansson',
            'Break Up (album)',
            'Anywhere I Lay My Head',
            'Lost in Translation (film)',
            'Girl with a Pearl Earring (film)'
        ]
    expected_scores = [1.0, 0.6042855, 0.5532534, 0.5457079, 0.5407755]

    result = subject.find_similar(
            vector,
            mention_text,
            use_fallback_for_empty_cs=False
        )[:5]

    result_ids = [r[0] for r in result]
    result_scores = [r[1] for r in result]

    for res, exp in zip(result_ids, expected_ids):
        assert res == exp, f"Result '{res}' != Expected '{exp}'"
    assert_almost_equal(result_scores, expected_scores, decimal=7)
