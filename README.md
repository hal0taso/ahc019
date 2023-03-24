# ahc019
AtCoder Heuristic Contest 019

## Problem
### Inpute
* D: integer
* f1: D \times D array representing the silhouette of G_1 in z-x plane 
* r1: D \times D array representing the silhouette of G_1 in z-y plane
* f2: D \times D array representing the silhouette of G_2 in z-x plane
* r2: D \times D array representing the silhouette of G_2 in z-y plane

### Idea
* Let v be an D \times D \times D array and (x, y, z) denoted by the (x*D^2 + y * D + z)-th elements of v;
* Let G_1, G_2 be two (undirected) graph whose vertex is v;
* f \colon G_1 \to G_2: permutation of v;

G_i is defined as 
* V(G_1) = v;
* E(G_1) = {{(x, y, z), (x+1, y, z)}: x \in [D-1]} \cup {{(x, y, z), (x, y+1, z)}: y \in [D-1]} \cup {{(x, y, z), (x, y, z+1)}: z \in [D-1]}
* V(G_2) = f(v);
* E(G_2) = similar to E(G_1)

f, rの制約を満たしているかどうか
-> f, rのテーブル持って、ブロックの数を足したり弾いたりする（更新毎に$O(V(C))$ Cは変更する連結成分だけかかる）

初期化パート
1. 各面で所望の影を構成する候補となるグラフを2つ（1つ目の影と2つめの影）作る
2. 初期状態では、全て孤立点の状態
3. まず、適当にG1から孤立点を選び、G2の頂点を選ぶ。
4. G1の変化に対して対応させる回転方向を決めて辺を伸ばしていく (孤立点ならマージしていく). このとき、辺の伸ばす方向は以下のような選び方しかできない（90度単位での回転）
  1. (dx, dy, dz) in G1 -> (dx, -dz, dy) in G2 (dxを軸とする正方向の90回転)
  2. (dx, dy, dz) in G1 -> (dx, dz, -dy) in G2 (dxを軸とする負方向の90回転)
  3. (dx, dy, dz) in G1 -> (dx, -dy, -dz) in G2 (dxを軸とする正方向の180回転)
  4. (dx, dy, dz) in G1 -> (-dz, dy, dx) in G2 (dyを軸とする正方向の90回転)
  5. (dx, dy, dz) in G1 -> (dz, dy, -dx) in G2 (dyを軸とする負方向の90回転)
  6. (dx, dy, dz) in G1 -> (-dx, dy, -dz) in G2 (dyを軸とする負方向の180回転)
  7. (dx, dy, dz) in G1 -> (-dy, dx, dz) in G2 (dzを軸とする正方向の90回転)
  8. (dx, dy, dz) in G1 -> (dy, -dx, dz) in G2 (dzを軸とする負方向の90回転)
  9. (dx, dy, dz) in G1 -> (-dx, -dy, dz) in G2 (dzを軸とする負方向の180回転)
  普通に軸固定して90度回転の行列かければ良い（3軸x3通り=9通りの回転）
5. 全頂点を起点に順に調べてゆき、それ以上伸ばせなくなったら初期化終了
