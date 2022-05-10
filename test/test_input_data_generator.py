import unittest
from configparser import ConfigParser
from torch import stack, equal, allclose, Tensor, LongTensor, Size
from src.input_data_generator import InputDataGenerator
from src.conll_document import ConllDocument
from src.toolbox import get_knowledge_base


class TestInputDataGenerator(unittest.TestCase):

    def setUp(self):
        self.config = ConfigParser()
        self.config.read('config.ini')

        self.knowledgebase = \
            get_knowledge_base(self.config, wiki='pedia', with_cg=False)

        tokenizer_pretrained_id = self.config['MODEL']['Model ID']
        self.subject = InputDataGenerator(
                knowledgebase=self.knowledgebase,
                tokenizer_pretrained_id=tokenizer_pretrained_id
            )

    def tearDown(self):
        # Get rid of test subject
        self.subject = None
        self.config = None

    def test_generate_for_conll_doc(self):
        """Tets generate input data for single CoNLL document"""
        doc = ConllDocument(
                """\
1	Germany\?\Germany 's\?\O representative\?\O to\?\O \
the\?\O European\?\European_Union Union\?\I 's\?\O veterinary\?\O \
committee\?\O Werner\?\B Zwingmann\?\I said\?\O on\?\O Wednesday\?\O \
consumers\?\O should\?\O buy\?\O sheepmeat\?\O"""
            )
        result_data, result_pos = self.subject.generate_for_conll_doc(doc)

        # Tokenized document
        expected_0 = [
            101, 2762, 1005, 1055, 4387, 2000, 1996, 2647, 2586, 1005,
            1055, 15651, 2837, 14121, 1062, 9328, 5804, 2056, 2006, 9317,
            10390, 2323, 4965, 8351, 4168, 4017, 102]

        seq_len = len(expected_0)
        # Attention mask
        expected_1 = [True] * seq_len

        # Padding
        expected_1 += [False] * (512 - seq_len)
        expected_0 += [False] * (512 - seq_len)

        # Token Type IDS (all 0; only one sequence)
        expected_2 = [False] * 512

        # Convert to tensors with shape (1, 512)
        expected_0 = LongTensor(expected_0).unsqueeze(0)
        expected_1 = LongTensor(expected_1).unsqueeze(0)
        expected_2 = LongTensor(expected_2).unsqueeze(0)

        # Expected MD labels
        expected_3 = [
                'None', 'B', 'O', 'None', 'O', 'O', 'O', 'B', 'I', 'O',
                'None', 'O', 'O', 'B', 'I', 'None', 'None', 'O', 'O', 'O',
                'O', 'O', 'O', 'O', 'None', 'None', 'None']
        expected_3 += ['None'] * (512 - seq_len)
        expected_3 = LongTensor(
                [self.subject.IOB_LABEL[ll] for ll in expected_3]
            ).unsqueeze(0)

        # Expected ED vectors
        expected_4 = [self.subject.EMPTY_EMBEDDING]
        expected_4 += [
                Tensor(self.knowledgebase.get_entity_vector('Germany'))
            ]
        expected_4 += [self.subject.EMPTY_EMBEDDING] * 5
        expected_4 += [
                Tensor(self.knowledgebase.get_entity_vector('European Union'))
            ]
        expected_4 += [self.subject.EMPTY_EMBEDDING] * (512 - len(expected_4))
        expected_4 = stack(expected_4).unsqueeze(0)

        expected_4_vector_size = self.subject.EMPTY_EMBEDDING.shape[0]
        expected_4_size = Size((1, 512, expected_4_vector_size))

        expected_5 = [None] * 512
        expected_5[1] = 'Germany'
        expected_5[7] = 'European_Union'

        # End position is length of tokenized document without [CLS] and [SEP]
        expected_pos = [(0, seq_len - 2)]

        self.assertEqual(len(result_data), len(result_pos))
        self.assertEqual(result_pos, expected_pos)

        self.assertEqual(len(result_data), 1)
        self.assertEqual(len(result_data[0]), 6)
        self.assertEqual(result_data[0][0].shape, Size((1, 512)))
        self.assertEqual(result_data[0][1].shape, Size((1, 512)))
        self.assertEqual(result_data[0][2].shape, Size((1, 512)))
        self.assertEqual(result_data[0][3].shape, Size((1, 512)))
        self.assertEqual(result_data[0][4].shape, expected_4_size)

        self.assertTrue(equal(result_data[0][0], expected_0))
        self.assertTrue(equal(result_data[0][1], expected_1))
        self.assertTrue(equal(result_data[0][2], expected_2))
        self.assertTrue(equal(result_data[0][3], expected_3))
        self.assertTrue(allclose(result_data[0][4], expected_4))
        self.assertEqual(expected_5, result_data[0][5])

    def test_generate_for_file(self):
        """Test generating for all CoNLL docs read from file"""
        conll_file = self.config['DATA']['Annotated Dataset']
        result = self.subject.generate_for_file(conll_file, progress=False)

        emb_size = self.subject.EMPTY_EMBEDDING.shape[0]
        # 1562 is the expected number of split sequences for AIDA-CoNLL docs
        self.assertEqual(result[0].shape, Size([1562, 512]))
        self.assertEqual(result[1].shape, Size([1562, 512]))
        self.assertEqual(result[2].shape, Size([1562, 512]))
        self.assertEqual(result[3].shape, Size([1562, 512]))
        self.assertEqual(result[4].shape, Size([1562, 512, emb_size]))
        self.assertEqual(len(result[5]), 1562)
        self.assertEqual(len(result[6]), 1562)
