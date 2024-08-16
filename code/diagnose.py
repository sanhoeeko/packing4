import gc
import os
import sys
import tracemalloc

import psutil


class MemoryMonitor:
    def __init__(self):
        self.previous_snapshot = self.getObjectsSnapshot()
        self.process = psutil.Process(os.getpid())
        self.previous_memory = self.process.memory_info().rss
        self.previous_python_memory = self.getPythonMemory()
        tracemalloc.start()
        self.previous_malloc = tracemalloc.take_snapshot()

    def getObjectsSnapshot(self) -> dict[str, list[int]]:
        """
        output format: { <type>: [number, total size] }
        """
        gc.collect()
        all_objects = gc.get_objects()
        class_count = {}
        for obj in all_objects:
            class_name = type(obj).__name__
            size = sys.getsizeof(obj)
            if class_name in class_count:
                class_count[class_name][0] += 1
                class_count[class_name][1] += size
            else:
                class_count[class_name] = [1, size]
        return class_count

    def getObjectsIncrease(self) -> dict[str, list[int]]:
        """
        output format: { <type>: [number, total size] }
        """
        current_snapshot = self.getObjectsSnapshot()
        new_objects = {}
        for class_name, (count, size) in current_snapshot.items():
            if class_name in self.previous_snapshot:
                new_count = count - self.previous_snapshot[class_name][0]
                new_size = size - self.previous_snapshot[class_name][1]
            else:
                new_count = count
                new_size = size
            if new_count > 0:
                new_objects[class_name] = [new_count, new_size]
        self.previous_snapshot = current_snapshot
        return new_objects

    def getMemoryGrowth(self) -> int:
        current_memory = self.process.memory_info().rss
        memory_growth = current_memory - self.previous_memory
        self.previous_memory = current_memory
        return memory_growth

    def getPythonMemory(self):
        gc.collect()
        all_objects = gc.get_objects()
        total_memory = sum(sys.getsizeof(obj) for obj in all_objects)
        return total_memory

    def getPythonMemoryGrowth(self):
        current_total_python_memory = self.getPythonMemory()
        python_memory_growth = current_total_python_memory - self.previous_python_memory
        self.previous_python_memory = current_total_python_memory
        return python_memory_growth

    def report(self):
        print(f"Memory growth: {self.getMemoryGrowth()}, in Python: {self.getPythonMemoryGrowth()}")
        print(self.getObjectsIncrease())
        print()

    def tracemallocReport(self, threshold=1024 * 1024):
        snapshot = tracemalloc.take_snapshot()
        total_memory = sum(stat.size for stat in snapshot.statistics('lineno'))
        if total_memory > threshold:
            top_stats = snapshot.statistics('lineno')
            print(f"[ Top 10 ] total: {total_memory / 1024} MB")
            for stat in top_stats[:10]:
                print(stat)
        self.previous_malloc = snapshot

    def tracemallocDiffReport(self, threshold=1024):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.compare_to(self.previous_malloc, 'lineno')
        total_leaked = sum(stat.size_diff for stat in top_stats)
        if total_leaked > threshold:
            print(f"[ Top 10 ] leaked: {total_leaked / 1024} KB")
            for stat in top_stats[:10]:
                print(stat)
        self.previous_malloc = snapshot
