import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "combined_repair_sr2.0" / "combined_repair_sr_optimized.py"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("combined_repair_sr_optimized", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ClassicPipelineTest(unittest.TestCase):
    def test_classic_pipeline_creates_repaired_and_enhanced_outputs(self):
        pipeline = load_pipeline_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "input.png"
            mask_path = tmp_path / "mask.png"
            output_dir = tmp_path / "out"

            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[:, :, 0] = np.linspace(20, 220, 32, dtype=np.uint8)
            image[:, :, 1] = np.linspace(30, 180, 24, dtype=np.uint8)[:, None]
            image[:, :, 2] = 120
            image[8:16, 12:20] = 0

            mask = np.zeros((24, 32), dtype=np.uint8)
            mask[8:16, 12:20] = 255

            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)

            config = pipeline.ModelConfig()
            config.backend = "classic"
            config.scale = 2
            config.validate()

            processor = pipeline.CombinedProcessor(config)
            self.assertEqual(processor.load_models("", ""), (True, True))

            success, result = processor.process_image(str(image_path), str(mask_path), str(output_dir))
            self.assertTrue(success, result)

            repaired = output_dir / "input_repaired.png"
            enhanced = Path(result)
            self.assertTrue(repaired.exists())
            self.assertTrue(enhanced.exists())
            with Image.open(enhanced) as output_image:
                self.assertEqual(output_image.size, (64, 48))


if __name__ == "__main__":
    unittest.main()
