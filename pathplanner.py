import itertools

class Node:
    '''
    Node Class for A* search data
    '''
    def __init__(self, spos, parent=None):
        self.spos = spos  # space position
        self.x = spos[0]
        self.y = spos[1]
        self.t = 0 if parent == None else parent.t + 1
        self.pos = (self.x, self.y, self.t)  # space-time position
        
        self.parent = parent

        self.g = 0  # cost from start to current
        self.h = 0  # heucost from current to end
        self.f = 0  # g + h

    def __eq__(self, other):
        return self.pos == other.pos

def stmap(_map, d):
    '''
    Convert space map to space-time map. Every obstacle is assumed to be fixed through time.
    '''
    pos = list(itertools.chain.from_iterable(itertools.repeat(x, d) for x in _map.keys()))
    time_col = list(range(d)) * len(_map.keys()) 
    new_pos = [(p[0],p[1], t) for p, t in zip(pos, time_col)]  # list of new space-time positions
    st_map = {}
    for p in new_pos:
        st_map[p] = _map[(p[0], p[1])]  # grab values for each space-time position from the space map given
    return st_map

def return_path(currentnode):
    '''
    Gets the return path from the currentnode.
    '''
    path = []
    current = currentnode
    while current != None:
        path.append(current.spos)
        current = current.parent
    path = path[::-1]
    return path

def search(st_map, start, end):
    '''
    A* algorithm with manhattan distance, used to calculate the best path for a single unit from start to end cell. 
    The provided _map should be a spacetime map.

    TODO: add a opponent unit iteraction strategy. right now assumes enemy units will remain stationary.
    can add a avoidance tendency for cells around the opponent unit. (soft avoid)
    can change cells around a given opponent cell to be impassable (hard avoid)
    can predict movement of opponent cell (too complicated for now, think more later)
    '''
    heu_to_end = lambda x: abs(x[0] - end[0]) + abs(x[1] - end[1])  # manhattan distance (L1 norm)
    add = lambda x, y: (x[0] + y[0], x[1] + y[1]) # tuple elementwise addition
    fixedcost = 1 # every step has the same distance for now.
    max_iter = 1000000
    iter_counter = 0

    start_node = Node(start)
    end_node = Node(end)
    openlist = {(start[0], start[1], 0):start_node}
    closedlist = {}

    adj_dirs = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]  # directions to adjacent cells

    while len(openlist) > 0:
        iter_counter += 1
        if iter_counter > max_iter:
            print('Max iterations reached. Search will quit.')
            break
        currentnode = None
        for node in openlist.values():
            if currentnode == None or node.f < currentnode.f:
                currentnode = node
        if currentnode == None:
            print('Path does not exist')
            return None
        if currentnode.spos == end_node.spos:  # if the space position is the same as the ending node
            path = return_path(currentnode)
            return path

        # remove from openlist and add to closedlist
        del openlist[currentnode.pos]
        closedlist[currentnode.pos] = currentnode
        
        # generate children
        children = []
        for d in adj_dirs:
            new_pos = add(currentnode.pos, d)
            st_pos = (new_pos[0], new_pos[1], currentnode.t + 1)
            if st_pos not in st_map:
                continue
            if st_map[st_pos] != 0:  # if impassable or reserved
                continue
            children.append(Node(new_pos, currentnode))

        for child in children:
            if child in closedlist.values():
                continue

            child.g = currentnode.g + fixedcost
            child.h = heu_to_end(child.pos)
            child.f = child.g + child.h

            if child in openlist.values() and child.g > openlist[child.pos].g:  
                # if already in openlist and the g value in openlist is smaller
                continue
            openlist[child.pos] = child

def main(_map, goals, d=5):
    '''
    goals input will specify which units wants to go where: List[(unit_id, start, end)]

    returns a actions Dict{unit_id:direction}
    '''
    pass