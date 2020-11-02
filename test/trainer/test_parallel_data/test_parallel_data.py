import unittest


class TestParallelData(unittest.TestCase):
    def test_run(self):
        from test.trainer.test_parallel_data import Pipeline
        from tfaip.base.data.data import DataBase, DataBaseParams

        class TestData(DataBase):
            def _get_train_data(self):
                pipeline = Pipeline(self, 8, 1000)
                return pipeline.output_generator()

            def _get_val_data(self):
                pass

            def _input_layer_specs(self):
                pass

            def _target_layer_specs(self):
                pass

        params = DataBaseParams()
        data = TestData(params)
        with data:
            for i, d in enumerate(zip(data._get_train_data(), range(100))):
                pass

