import unittest
import pandas as pd
import numpy as np
from src.scoring import calculate_tag_scores
from src.clustering import assign_primary_cluster, generate_secondary_clusters
from src.analysis import calculate_representativeness

class TestClusteringPipeline(unittest.TestCase):
    def setUp(self):
        self.texts = ["This is about money and economy.", "This is playing soccer."]
        self.tags = ["Economy", "Sports"]
        
    def test_scoring_shape(self):
        df, embeddings = calculate_tag_scores(self.texts, self.tags)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(len(embeddings), 2)
        # Check if score for economy is higher for first text
        self.assertGreater(df.iloc[0]['Economy'], df.iloc[0]['Sports'])

    def test_clustering(self):
        # Mock scores
        scores = pd.DataFrame({
            'Economy': [0.9, 0.1],
            'Sports': [0.1, 0.9]
        })
        primary = assign_primary_cluster(scores)
        self.assertEqual(primary[0], 'Economy')
        self.assertEqual(primary[1], 'Sports')

    def test_secondary_clustering(self):
        # Mock data
        df = pd.DataFrame({'primary': ['A', 'A', 'A']})
        embeddings = np.random.rand(3, 10)
        # 3 items, 2 clusters -> should work
        secondary = generate_secondary_clusters(df, embeddings, 'primary', n_clusters=2)
        self.assertEqual(len(secondary), 3)

if __name__ == '__main__':
    unittest.main()
