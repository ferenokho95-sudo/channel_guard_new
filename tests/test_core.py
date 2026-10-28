import tempfile
import unittest
from pathlib import Path
from guard.data import split, load, write
from guard.generate import generate
from guard.text import ATTACKS, augment, normalize, transform
from guard.experiment import select_threshold, metrics

class CoreTests(unittest.TestCase):
    def test_no_family_leakage(self):
        parts = split(generate())
        sets = [set(r['group_id'] for r in part) for part in parts]
        for i in range(3):
            self.assertEqual({r['label'] for r in parts[i]}, {0,1})
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])
        self.assertEqual(sum(map(len, parts)), len(generate()))
    def test_roundtrip_and_review_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.csv'; rows = generate(); rows[0]['review_status']='approved'
            write(path, rows)
            self.assertEqual(len(load(path)),640)
            self.assertEqual(len(load(path, True)),1)
    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.csv'; row=generate()[0]; write(path,[row,row])
            with self.assertRaises(ValueError):
                load(path)
    def test_rejected_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.csv'; rows=generate(); rows[0]['review_status']='rejected'
            write(path,rows)
            self.assertEqual(len(load(path)),639)
    def test_augmentation_keeps_lineage(self):
        row=generate()[0]; result=augment([row]); self.assertEqual(len(result),6)
        self.assertTrue(all(r['label']==row['label'] and r['group_id']==row['group_id'] for r in result))
        for attack in ATTACKS:
            self.assertIsInstance(transform('Напиши @example.user',attack),str)
    def test_normalization(self):
        self.assertEqual(normalize('t\u200b.me [собака] user'), 't.me@user')
    def test_threshold(self):
        t=select_threshold([0,1],[.2,.4]); self.assertEqual(metrics([0,1],[.2,.4],t)['f1'],1.)

if __name__ == '__main__':
    unittest.main()
