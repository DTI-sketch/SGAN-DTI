import unittest

import dgl
import numpy as np
import torch

from utils.load_graph import assert_training_only_dti, remove_heldout_dti_edges


class LeakageRemovalTest(unittest.TestCase):
    def test_validation_and_test_edges_are_removed_in_both_directions(self):
        all_pairs = np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
        graph = dgl.heterograph(
            {
                ("drug", "d_p", "target"): (
                    torch.from_numpy(all_pairs[0]),
                    torch.from_numpy(all_pairs[1]),
                ),
                ("target", "p_d", "drug"): (
                    torch.from_numpy(all_pairs[1]),
                    torch.from_numpy(all_pairs[0]),
                ),
            },
            num_nodes_dict={"drug": 3, "target": 3},
        )
        train = all_pairs[:, :1]
        validation = all_pairs[:, 1:2]
        test = all_pairs[:, 2:]

        remove_heldout_dti_edges(graph, validation, test)
        assert_training_only_dti(graph, train, validation, test, "toy_graph")
        self.assertEqual(graph.num_edges("d_p"), 1)
        self.assertEqual(graph.num_edges("p_d"), 1)


if __name__ == "__main__":
    unittest.main()
