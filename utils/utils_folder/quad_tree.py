class Point:
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y


class Node:
    def __init__(self, pos: Point = None, data: float = 0.0):
        self.pos = pos
        self.data = data


class Quad:
    def __init__(self, top_l: Point = Point(), bot_R: Point = Point()):
        self.top_left = top_l
        self.bot_right = bot_R

        self.width = self.top_left.x + self.bot_right.x
        self.height = self.top_left.y + self.bot_right.y

        self.n = Node()

        self.top_left_tree = None
        self.top_right_tree = None
        self.bot_left_tree = None
        self.bot_right_tree = None

    def insert(self, node: Node):
        
        if abs(self.top_left.x - self.bot_right.x) <= 1 and abs(self.top_left.y - self.bot_right.y <= 1):
            if self.n.pos is None:
                self.n = node
            return
        
        if int(self.width / 2) >= node.pos.x:
            # top left tree
            if int(self.height / 2) >= node.pos.y:
                if self.top_left_tree is None:
                    self.top_left_tree = Quad(
                        Point(self.top_left.x, self.top_left.y), 
                        Point(int(self.width / 2), int(self.height / 2))
                    )
                
                self.top_left_tree.insert(node)
            # bot left tree
            else:
                if self.bot_left_tree is None:
                    self.bot_left_tree = Quad(
                        Point(self.top_left.x, int(self.height / 2)),
                        Point(int(self.width / 2), self.bot_right.y)
                    )
                    
                self.bot_left_tree.insert(node)
                
        else:
            #top right tree
            if int(self.height / 2) >= node.pos.y:
                if self.top_right_tree is None:
                    self.top_right_tree = Quad(
                        Point(int(self.width / 2), self.top_left.y),
                        Point(self.bot_right.x, int(self.height / 2))
                    )
                    
                self.top_right_tree.insert(node)
                
            else:
                if self.bot_right_tree is None:
                    self.bot_right_tree = Quad(
                        Point(int(self.width / 2),
                          int(self.height / 2)),
                        Point(self.bot_right.x, self.bot_right.y)
                    )
                    
                self.bot_right_tree.insert(node)
                
    def search(self, p: Point):
        if self.n.pos != None:
            return self.n
        
        if int(self.width / 2) >= p.x:
            # top left tree
            if int(self.height / 2) >= p.y:
                if self.top_left_tree is None:
                    return None
                return self.top_left_tree.search(p)
            # bot left tree
            else:
                if self.bot_left_tree is None:
                    return None
                return self.bot_left_tree.search(p)
            
        else:
            # top right tree
            if int(self.height / 2) >= p.y:
                if self.top_right_tree is None:
                    return None
                return self.top_right_tree.search(p)
            # bot right tree
            else:
                if self.bot_right_tree is None:
                    return None
                return self.bot_right_tree.search(p)

    def findContainingQuad(self, t_left: Point, b_right: Point):    
        if t_left.x <= self.top_left.x and t_left.y <= self.top_left.y and b_right.x >= self.bot_right.x and b_right.y >= self.bot_right.y:
            return None
        
        q = None
        
        # left_tree
        if int(self.width / 2) >= b_right.x:
            # top left tree
            if int(self.height / 2) >= b_right.y:
                if self.top_left_tree is None:
                    return None
                q = self.top_left_tree.findContainingQuad(t_left, b_right)
            # bot left tree
            else:
                if self.bot_left_tree is None:
                    return None
                q = self.bot_left_tree.findContainingQuad(t_left, b_right)
        else:
            # top right tree
            if int(self.height / 2) >= b_right.y:
                if self.top_right_tree is None:
                    return None
                q = self.top_right_tree.findContainingQuad(t_left, b_right)
            # bot left tree
            else:
                if self.bot_right_tree is None:
                    return None
                q = self.bot_right_tree.findContainingQuad(t_left, b_right)

        if q is None:
            return self

        return q
            
    def gather(self):
        nodes = []

        if self.n.pos != None:
            return [self.n]
        
        if self.top_left_tree is not None:
            node = self.top_left_tree.gather()
            nodes = nodes + node

        if self.bot_left_tree is not None:
            node = self.bot_left_tree.gather()
            nodes = nodes + node

        if self.top_right_tree is not None:
            node = self.top_right_tree.gather()
            nodes = nodes + node

        if self.bot_right_tree is not None:
            node = self.bot_right_tree.gather()
            nodes = nodes + node
            
        return nodes
            
    def findInRadius(self, p: Point, radius):
        min_x = max(p.x - radius, 0)
        max_x = min(p.x + radius, self.width)
        
        min_y = max(p.y - radius, 0)
        max_y = min(p.y + radius, self.height)
        
        quad = self.findContainingQuad(Point(min_x, min_y), Point(max_x, max_y))
        
        if quad is None:
            return []
        
        nodes = quad.gather()
                            
        return nodes
