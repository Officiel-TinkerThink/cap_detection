import os
import tempfile
from pathlib import Path
import pytest
from scripts.adjust_label import Relabeler

@pytest.fixture
def setup_test_environment():
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = Path(temp_dir) / "source"
        target_dir = Path(temp_dir) / "target"
        
        # Create source directory and test files
        source_dir.mkdir()
        target_dir.mkdir()
        
        # Create test label files
        test_cases = [
            ("img_b2_001.txt", "0 0.5 0.5 0.1 0.1\n0 0.6 0.6 0.1 0.1"),
            ("img_b3_002.txt", "0 0.5 0.5 0.1 0.1"),
            ("img_b4_003.txt", "0 0.7 0.7 0.1 0.1\n0 0.8 0.8 0.1 0.1"),
        ]
        
        for filename, content in test_cases:
            (source_dir / filename).write_text(content)
            
        # Create corresponding image files (empty)
        for filename, _ in test_cases:
            (source_dir / filename.replace(".txt", ".jpg")).touch()
        
        yield source_dir, target_dir
        # Teardown is handled by the TemporaryDirectory context manager

def test_relabeler_classify(setup_test_environment):
    source_dir, target_dir = setup_test_environment
    code2idx = {"b2": 2, "b3": 2, "b4": 0}
    relabeler = Relabeler(code2idx, str(source_dir), str(target_dir))
    
    # Test classification for each file
    for filename in source_dir.glob("*.txt"):
        assert relabeler.classify(filename.name)
        
        # Verify the output file was created
        output_file = target_dir / filename.name
        assert output_file.exists()
        
        # Verify the content was modified correctly
        content = output_file.read_text()
        first_char = content[0]
        assert first_char in ['0', '2']  # Based on code2idx mapping

def test_relabeler_process(setup_test_environment):
    source_dir, target_dir = setup_test_environment
    code2idx = {"b2": 2, "b3": 2, "b4": 0}
    relabeler = Relabeler(code2idx, str(source_dir), str(target_dir))
    
    # Process all files
    relabeler.process()
    
    # Verify all files were processed
    source_files = list(source_dir.glob("*.txt"))
    target_files = list(target_dir.glob("*.txt"))
    assert len(source_files) == len(target_files)
    
    # Verify content of one file
    test_file = "img_b2_001.txt"
    output_content = (target_dir / test_file).read_text()
    assert output_content.startswith("2 ")  # b2 should be mapped to 2

def test_relabeler_invalid_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        relabeler = Relabeler({}, temp_dir, temp_dir)
        # Test with non-existent file
        assert not relabeler.classify("nonexistent.txt")