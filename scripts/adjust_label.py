import glob
from pathlib import Path
import re

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

    def classify(self, file_name:str) -> None:
        """
        Classify the label file
        
        Args:
          file_name (str): the name of the label file
        """
        match = re.search(r"_b(\d+)_", file_name)
        code = match.group(1)
        idx = self.code2idx[code]
        with open(Path(self.source, file_name), "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()
      
        modified_lines = []
        # changed the first char of every lines
        for line in lines:
            if line.strip():
                modified_line = str(idx) + line[1:]
                modified_lines.append(modified_line)

        with open(Path(self.target, file_name), 'w') as f_out:
          f_out.writelines(modified_lines)  

    def process(self) -> None:
        """
        Process the label files
        """
        for im_path in sorted(glob.glob(f"{self.source}/*.jpg")):
            label_file = Path(im_path.replace(".jpg", ".txt")).name
            self.classify(label_file)


if __name__ == "__main__":
    code2idx = {"2":2, "3":2, "4":0, "5":1}
    labeler = Relabeler(code2idx, 'sample', 'sample')
    labeler.process()
