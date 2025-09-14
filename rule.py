
def is_inside(x: float, y: float,) -> bool:
    left = 0.25
    right = 0.75
    bottom = 0.25
    top = 0.75

    if left <= x <= right and bottom <= y <= top:
        return 1
    else:
        return 0
