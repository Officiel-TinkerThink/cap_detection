import os
import tempfile
from pathlib import Path

import pytest

from scripts.dataset import DataSplitter


@pytest.fixture
def setup_test_environment():
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = Path(temp_dir) / "source"
        target_dir = Path(temp_dir) / "target"

        # Create source directory
        source_dir.mkdir()

        # Create test files
        test_cases = [
            ("img_b2_001.jpg", "img_b2_001.txt", "0 0.5 0.5 0.1 0.1"),
            ("img_b3_002.jpg", "img_b3_002.txt", "0 0.5 0.5 0.1 0.1"),
            ("img_b4_003.jpg", "img_b4_003.txt", "0 0.7 0.7 0.1 0.1"),
            ("img_b5_004.jpg", "img_b5_004.txt", "0 0.8 0.8 0.1 0.1"),
        ]

        # Create test files
        for img_name, txt_name, content in test_cases:
            (source_dir / img_name).touch()
            (source_dir / txt_name).write_text(content)

        yield source_dir, target_dir
        # Teardown is handled by the TemporaryDirectory context manager


def test_initialization(setup_test_environment):
    source_dir, target_dir = setup_test_environment
    splitter = DataSplitter(str(source_dir), str(target_dir))

    # Check if target directories were created
    assert (target_dir / "images" / "train").exists()
    assert (target_dir / "images" / "val").exists()
    assert (target_dir / "labels" / "train").exists()
    assert (target_dir / "labels" / "val").exists()


def test_mapping(setup_test_environment):
    source_dir, target_dir = setup_test_environment
    splitter = DataSplitter(str(source_dir), str(target_dir))

    # Call the mapping method
    splitter.mapping()

    # Check if files are correctly mapped by code
    assert len(splitter.bucket) > 0
    for code, files in splitter.bucket.items():
        assert code in ["b2", "b3", "b4", "b5"]
        assert all(f"_{code}_" in f for f in files)


def test_split_distribution(setup_test_environment):
    source_dir, target_dir = setup_test_environment

    # Add more test files to ensure we have multiple instances per class
    for i in range(2, 5):  # Add more instances for each class
        for code in ["b2", "b3", "b4", "b5"]:
            img_name = f"img_{code}_00{i}.jpg"
            txt_name = f"img_{code}_00{i}.txt"
            (source_dir / img_name).touch()
            (source_dir / txt_name).write_text("0 0.5 0.5 0.1 0.1")

    splitter = DataSplitter(str(source_dir), str(target_dir), train_ratio=0.7)
    splitter.split(seed=42)

    # Count files
    train_imgs = len(list((target_dir / "images" / "train").glob("*.jpg")))
    val_imgs = len(list((target_dir / "images" / "val").glob("*.jpg")))
    train_labels = len(list((target_dir / "labels" / "train").glob("*.txt")))
    val_labels = len(list((target_dir / "labels" / "val").glob("*.txt")))

    # With more instances, we should get both train and val
    assert train_imgs > 0, "No training images found"
    assert val_imgs > 0, "No validation images found"
    assert train_imgs + val_imgs >= 4, "Not all images were processed"
    assert train_imgs == train_labels, "Mismatch between train images and labels"
    assert val_imgs == val_labels, "Mismatch between val images and labels"


def test_split_with_single_file_per_code(setup_test_environment):
    source_dir, target_dir = setup_test_environment
    # Create a test case with only one file per code
    test_cases = [
        ("single_b2.jpg", "single_b2.txt", "0 0.5 0.5 0.1 0.1"),
        ("single_b3.jpg", "single_b3.txt", "0 0.5 0.5 0.1 0.1"),
    ]

    for img_name, txt_name, content in test_cases:
        (source_dir / img_name).touch()
        (source_dir / txt_name).write_text(content)

    splitter = DataSplitter(str(source_dir), str(target_dir), train_ratio=0.5)
    splitter.split()

    # All single-instance codes should go to train
    train_imgs = len(list((target_dir / "images" / "train").glob("*.jpg")))
    assert train_imgs >= 2, "Single instance codes should go to train"
