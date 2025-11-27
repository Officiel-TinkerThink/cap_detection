import glob
import re
from pathlib import Path


class Relabeler:
    def __init__(self, code2idx: dict[str, int], source: str, target: str):
        """
        Initialization of Relabeler Class

        Args:
          code2idx (dict[str, int]): the mapping from code to index
          source (str): the path to the source directory
          target (str): the path to the target directory
        """
        self.code2idx = code2idx
        self.source = source
        self.target = target

        if self.target:
            Path(self.target).mkdir(parents=True, exist_ok=True)

    def classify(self, file_name: str) -> bool:
        """
        Classify the label file

        Args:
          file_name (str): the name of the label file

        Returns:
          bool: True
        """
        match = re.search(r"_(b\d)_", file_name)
        if not match:
            print(f"Warning: Could not extract code from filename: {file_name}")
            return False

        try:
            code = match.group(1)
            idx = self.code2idx[code]

            # Rest of your existing code...

        except (KeyError, FileNotFoundError) as e:
            print(f"Error processing {file_name}: {str(e)}")
            return False

        with open(Path(self.source, file_name), "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        modified_lines = []

        # change the first char of every lines
        for line in lines:
            if line.strip():
                modified_line = str(idx) + line[1:]
                modified_lines.append(modified_line)

        with open(Path(self.target, file_name), "w") as f_out:
            f_out.writelines(modified_lines)

        print(f"Relabelling {file_name} complete")
        return True

    def process(self) -> None:
        """
        Process the label files
        """
        count = 0
        for im_path in sorted(glob.glob(f"{self.source}/*.jpg")):
            label_file = Path(im_path.replace(".jpg", ".txt")).name
            count += int(self.classify(label_file))

        print(f"Relabelling {count} files complete")


if __name__ == "__main__":
    code2idx = {"b2": 2, "b3": 2, "b4": 0, "b5": 1}
    labeler = Relabeler(code2idx, "sample", "sample")
    labeler.process()
