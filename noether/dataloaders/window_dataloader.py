from collections import deque

class MovingWindowDataloader():
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = int(window_size) + 1 # len(window_size) for x and 1 for y
    
    def __iter__(self):
        """Yield batches of graphs from the iterable."""
        buffer = deque(maxlen=self.window_size)
        for graph in self.data:
            buffer.append(graph)
            if len(buffer) < self.window_size:
                continue
            
            data = list(buffer)
            x = data[:-1]
            y = data[-1]

            yield x, y
        
    def __len__(self):
        return max(len(self.data) - self.window_size + 1, 0)