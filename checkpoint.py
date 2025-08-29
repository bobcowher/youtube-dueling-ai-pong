import copy, random, torch
from collections import deque

from torch.nn import MultiLabelSoftMarginLoss

class CheckpointPool:
    def __init__(self, max_size=5) -> None:
        self.max_size = max_size
        self.pool: deque[tuple[float, torch.nn.Module]] = deque()

    def add(self, model, score):
        self.pool.append((score, copy.deepcopy(model)))
        self.pool = deque(sorted(self.pool, key=lambda x: x[0], reverse=True))

        while len(self.pool) > self.max_size:
            self.pool.pop()

    def sample(self):
        if not self.pool:
            raise RuntimeError("Checkpoint pool empty!")

        _, models = zip(*self.pool)
        return random.choice(models)


    def report(self):

        if not self.pool:
            print("CheckpointPool is empty.")
            return

        rows = []

        for rank, (score, model) in enumerate(self.pool, start=1):
            rows.append(f"|{rank:^7}|{score:^20.2f}")

        message = ("Rank - Skill\n"
               + "\n".join(rows) + "\n")
        
        print(message)


    def __len__(self):
        return len(self.pool)

    def best_score(self):
        return self.pool[0][0] if self.pool else None
