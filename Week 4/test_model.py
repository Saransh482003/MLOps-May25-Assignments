import unittest
import iris_model as model

class TestIrisDecisionTree(unittest.TestCase):
    def setUp(self):
        self.data = model.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = model.split_data(self.data)
        self.clf = model.train_model(self.X_train, self.y_train)

    def test_model_accuracy(self):
        acc, _ = model.evaluate_model(self.clf, self.X_test, self.y_test)
        print(f"\n✅ Accuracy: {acc:.3f}")
        self.assertGreaterEqual(acc, 0.90, f"Model accuracy too low: {acc}")

    def test_classification_report(self):
        _, report = model.evaluate_model(self.clf, self.X_test, self.y_test)
        print("\n📊 Classification Report:")
        print(report)
        self.assertIn("setosa", report)
        self.assertIn("versicolor", report)
        self.assertIn("virginica", report)

if __name__ == "__main__":
    unittest.main(verbosity=2)