from typing import List


def maximumWhiteTiles(tiles: List[List[int]], carpetLen: int) -> int:
    prefix = [0] * tiles[-1][-1]
    for s, e in tiles:
        prefix[s - 1:e] = [1] * (e + 1 - s)
    
    n = len(prefix)
    if carpetLen > n: return sum(prefix)
    
    max_ = cur_ = sum(prefix[:carpetLen])
    
    for i in range(carpetLen, len(prefix)):  # 滑动窗口
        
        cur_ -= prefix[i - carpetLen]
        cur_ += prefix[i]
        max_ = max(max_, cur_)
    
    return max_


print(maximumWhiteTiles(
    [[1, 5], [10, 11], [12, 18], [20, 25], [30, 32]], 10
))
