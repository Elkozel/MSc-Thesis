import pytest
from noether.utils.window_dataloader import MovingWindowDataloader

# ---------- TESTS ----------

def test_len_matches_expected():
    data = list(range(10))
    window_size = 3
    loader = MovingWindowDataloader(data, window_size)
    items = list(loader)
    assert len(loader) == 7  # 10 - 3
    assert len(items) == len(loader)

def test_yields_correct_number_of_batches():
    data = list(range(5))
    window_size = 2
    loader = MovingWindowDataloader(data, window_size)
    output = list(loader)
    assert len(output) == 3  # 5 - 2

def test_batch_contents_correct():
    data = list(range(5))
    window_size = 2
    loader = MovingWindowDataloader(data, window_size)
    output = list(loader)
    expected = [
        ([0, 1], 2),
        ([1, 2], 3),
        ([2, 3], 4)
    ]
    assert output == expected

def test_short_data_yields_one_record():
    data = [1, 2]
    window_size = 3
    loader = MovingWindowDataloader(data, window_size)
    output = list(loader)
    assert output == []

def test_window_size_one():
    data = [10, 20, 30]
    window_size = 1
    loader = MovingWindowDataloader(data, window_size)
    output = list(loader)
    expected = [
        ([10], 20),
        ([20], 30)
    ]
    assert output == expected

def test_empty_data():
    data = []
    window_size = 2
    loader = MovingWindowDataloader(data, window_size)
    assert list(loader) == []
    assert len(loader) == 0

def test_multiple_passes():
    data = [s for s in range(10)]
    window_size = 2
    loader = MovingWindowDataloader(data, window_size)

    first_pass = []
    for item in loader:
        first_pass.append(item)

    for idx, item in enumerate(loader):
        assert item == first_pass[idx]