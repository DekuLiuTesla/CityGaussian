import os.path
import unittest
from internal.dataparsers.colmap_dataparser import Colmap


class ColmapDataparserTestCase(unittest.TestCase):
    def test_eval_list(self):
        eval_list = os.path.expanduser("~/data/Mega-NeRF/rubble-pixsfm/val_set.txt")

        eval_set = {}
        with open(eval_list, "r") as f:
            for row in f:
                eval_set[row.rstrip("\n")] = True

        datapatser = Colmap(
            split_mode="experiment",
            eval_image_select_mode="list",
            eval_list=os.path.expanduser(eval_list),
        ).instantiate(os.path.expanduser("~/data/Mega-NeRF/rubble-pixsfm/colmap/"), os.getcwd(), 0)
        dataparser_outputs = datapatser.get_outputs()
        for i in dataparser_outputs.train_set.image_names:
            self.assertTrue(i not in eval_set)
        for i in dataparser_outputs.val_set.image_names:
            self.assertTrue(i in eval_set)

        datapatser = Colmap(
            split_mode="reconstruction",
            eval_image_select_mode="list",
            eval_list=os.path.expanduser(eval_list),
        ).instantiate(os.path.expanduser("~/data/Mega-NeRF/rubble-pixsfm/colmap/"), os.getcwd(), 0)
        dataparser_outputs = datapatser.get_outputs()
        for i in eval_set:
            dataparser_outputs.train_set.image_names.index(i)
        for i in dataparser_outputs.val_set.image_names:
            self.assertTrue(i in eval_set)


if __name__ == '__main__':
    unittest.main()
