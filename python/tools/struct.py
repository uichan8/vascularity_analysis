class vnode:
    def __init__(self,id_num):
        self.id_num = id_num
        self.head = []
        self.tail = []

    def change_head_tail(self):
        self.head, self.tail = self.tail, self.head
    
    def push_head(self,id):
        self.head.append(id)

    def push_tail(self,id):
        self.tail.append(id)
    
    def disconnect(self,id):
        try:
            self.head.remove(id)
        except:
            pass

        try:
            self.tail.remove(id)
        except:
            pass


class vbranch(vnode):
    def __init__(self,id):
        super().__init__(id)
        self.branch_poly_x = []
        self.branch_poly_y = []
        self.branch_poly_r = []
        self.end_points = []

class vbifur(   vnode):
    def __init__(self,id):
        super().__init__(id)
        self.center_coor = None
        self.vbifur_mask = None
        self.bifur_edge = None

class vgraph():
    def __init__(self):
        self.vbranches = []
        self.vbifers = []
    
    def connect(self,bifur_num,branch_num):
        pass