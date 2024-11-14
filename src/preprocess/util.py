from collections import Counter

def majority_element(nums):
    counts = Counter(nums)
    return max(counts, key=counts.get)