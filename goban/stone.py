BLACK = 1
WHITE = -1


class Stone:
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y
        self.group = None
    
    def __repr__(self):
        color = "Black" if self.color == BLACK else "White"
        return f"Color: {color} Coords: {self.x}, {self.y}"
