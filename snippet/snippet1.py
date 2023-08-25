#####################################################################
## Union Find

class UnionFind():
    # 初期化
    def __init__(self, n):
        self.par = [-1] * n
        self.rank = [0] * n
        self.siz = [1] * n

    # 根を求める
    def root(self, x):
        if self.par[x] == -1: return x # x が根の場合は x を返す
        else:
          self.par[x] = self.root(self.par[x]) # 経路圧縮
          return self.par[x]

    # x と y が同じグループに属するか (根が一致するか)
    def issame(self, x, y):
        return self.root(x) == self.root(y)

    # x を含むグループと y を含むグループを併合する
    def unite(self, x, y):
        # x 側と y 側の根を取得する
        rx = self.root(x)
        ry = self.root(y)
        if rx == ry: return False # すでに同じグループのときは何もしない
        # union by rank
        if self.rank[rx] < self.rank[ry]: # ry 側の rank が小さくなるようにする
            rx, ry = ry, rx
        self.par[ry] = rx # ry を rx の子とする
        if self.rank[rx] == self.rank[ry]: # rx 側の rank を調整する
            self.rank[rx] += 1
        self.siz[rx] += self.siz[ry] # rx 側の siz を調整する
        return True
    
    # x を含む根付き木のサイズを求める
    def size(self, x):
        return self.siz[self.root(x)]
    

###################################################################
## セグメントツリー

#####segfunc#####
def segfunc(x, y):
    return min(x, y) # 最小値を求めるときの例
#################

#####ide_ele#####
## モノイドの単位元を設定する
ide_ele = float('inf') # 最小値を求めるときの例
#################

class SegTree:
    """
    init(init_val, ide_ele): 配列init_valで初期化 O(N)
    update(k, x): k番目の値をxに更新 O(logN)
    query(l, r): 区間[l, r)をsegfuncしたものを返す O(logN)
    """
    def __init__(self, init_val, segfunc, ide_ele):
        """
        init_val: 配列の初期値
        segfunc: 区間にしたい操作
        ide_ele: 単位元
        n: 要素数
        num: n以上の最小の2のべき乗
        tree: セグメント木(1-index)
        """
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.tree = [ide_ele] * 2 * self.num
        # 配列の値を葉にセット
        for i in range(n):
            self.tree[self.num + i] = init_val[i]
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = self.segfunc(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, k, x):
        """
        k番目の値をxに更新
        k: index(0-index)
        x: update value
        """
        k += self.num
        self.tree[k] = x
        while k > 1:
            self.tree[k >> 1] = self.segfunc(self.tree[k], self.tree[k ^ 1])
            k >>= 1

    def query(self, l, r):
        """
        [l, r)のsegfuncしたものを得る
        l: index(0-index)
        r: index(0-index)
        """
        res = self.ide_ele

        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res = self.segfunc(res, self.tree[l])
                l += 1
            if r & 1:
                res = self.segfunc(res, self.tree[r - 1])
            l >>= 1
            r >>= 1
        return res



######################################################################
## エラトステネスの篩

def eratosthenes(n):
    # n までの自然数を列挙する
    isPrimes = [True] * (n+1)

    # 0 と 1 を取り除く
    isPrimes[0], isPrimes[1] = False, False

    # 2 から √n まで繰り返す
    for i in range(2, int(n**0.5)+1):
        # i が取り除かれていないとき
        if isPrimes[i]:
            # i の倍数を取り除く
            for j in range(2*i, n+1, i):
                isprimes[j] = False

    # 2 から n までの素数のリストを返す
    return [i for i in range(2, n+1) if isPrimes[i]]


###################################################################
## 素数判定

import math

def is_prime(num):
    if num < 2:
        return False
    elif num == 2:
        return True
    elif num % 2 == 0:
        return False
    
    sqrtNum = math.floor(math.sqrt(num))
    for i in range(3, sqrtNum + 1, 2):
        if num % i == 0:
            return False
    
    return True


##################################################################
## 素因数分解（素因数と指数の組みのリスト）

def factorization(n):
    arr = []
    temp = n
    for i in range(2, int(-(-n**0.5//1))+1):
        if temp%i==0:
            cnt=0
            while temp%i==0:
                cnt+=1
                temp //= i
            arr.append([i, cnt])

    if temp!=1:
        arr.append([temp, 1])

    if arr==[]:
        arr.append([n, 1])

    return arr


###################################################################
## 約数列挙

def divisor(n):
    ds = []
    for i in range(1, int(n ** 0.5) + 1):
        # nをiで割り切れる場合
        if n % i == 0:
            # iは約数
            ds.append(i)
            # n = i^2ではない場合
            if i != n // i:
                ds.append(n//i)
    ds.sort()
    return ds 

###############################################################
## typical bfs

from collections import deque

n, m = map(int, input().split())

graph = [[] for _ in range(n+1)]

for i in range(m):
 a, b = map(int, input().split())
 graph[a].append(b)
 graph[b].append(a)

dist = [-1] * (n+1)
dist[0] = 0
dist[1] = 0

d = deque()
d.append(1)

while d:
 v = d.popleft()
 for i in graph[v]:
   if dist[i] != -1:
     continue
   dist[i] = dist[v] + 1
   d.append(i)

ans = dist[1:]
print(*ans, sep="\n")

#################################################################
## typical dfs

import sys
sys.setrecursionlimit(10**7) # 再起回数の設定

H, W = map(int, input().split())
maze = [list(input()) for h in range(H)]

for h in range(H):
    for w in range(W):
        if maze[h][w] == "s":
            sx, sy = h, w

# 深さ優先探索
def dfs(x, y):
    # 範囲外や壁の場合は終了
    if y >= W or y < 0 or x >= H or x < 0 or maze[x][y] == '#':
        return

    # ゴールに辿り着ければ終了
    if maze[x][y] == 'g':
        print('Yes')
        exit()

    maze[x][y] = '#' # 確認したルートは壁にしておく

    # 上下左右への移動パターンで再起していく
    dfs(x+1, y)
    dfs(x-1, y)
    dfs(x, y+1)
    dfs(x, y-1)

dfs(sx, sy) # スタート位置から深さ優先探索
print('No')


############################################################
## 循環小数の計算

def unitFraction(numelator, duplicator):
    num = []
    set_of_numelators = set()
    while numelator%duplicator != 0:
        num.append(numelator//duplicator)
        numelator = (numelator%duplicator) * 10
        if numelator not in set_of_numelators:
            set_of_numelators.add(numelator)
        elif numelator in set_of_numelators:
            return num
        elif numelator%duplicator == 0:
            return num
        
##########################################################
## 循環検出

def findDuplicate(nums):
    hare = tortoise = 0
    if nums == None:
        return 0
    # カメは速度1, ウサギは速度2で走る
    while True: # ぶつかるまでループ
        tortoise = nums[tortoise]
        hare = nums[hare]
        hare = nums[hare]
        if tortoise == hare:
            break
    m = tortoise
    # ウサギを初期位置に戻す
    hare = 0
    while tortoise != hare: # ぶつかるまでループ
        tortoise = nums[tortoise]
        hare = nums[hare] # ウサギも速度1
    l = hare
    u = 0
    # ループ内で重なっている位置からウサギだけを動かす
    while True: # ぶつかるまでループ
        u += 1
        hare = nums[hare]
        hare = nums[hare]
        if tortoise == hare:
            break
    # m:衝突が発生するまでの階数
    # l:、u:ループの長さ
    return l

##################################################################
## 