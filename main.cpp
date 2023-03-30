#include <bits/stdc++.h>
#include <sys/time.h>

using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef long double ld;
#define FOR(i, a, b) for (ll i = a; i < b; ++i)
#define REP(i, n) for (ll i = 0; i < n; ++i)
#define all(...) std::begin(__VA_ARGS__), std::end(__VA_ARGS__)
#define rall(...) std::rbegin(__VA_ARGS__), std::rend(__VA_ARGS__)

using vvvvi = vector<vector<vector<vector<int>>>>;
using vvvi = vector<vector<vector<int>>>;
using vvi = vector<vector<int>>;
using vi = vector<int>;
random_device seed_gen;
mt19937 engine(seed_gen());
// mt19937 engine(0);
constexpr long long TIME_LIMIT = 5000;
constexpr long long TIME_LIMIT_M = 3000;
constexpr long long TIME_LIMIT_SA = 2000;
constexpr long long INF = 1000000000LL;

map<pair<int, int>, unsigned long long> hashtable;

struct UnionFind
{
    std::vector<int> par; // par[i]:iの親の番号　(例) par[3] = 2 : 3の親が2
    vector<int> _size;
    UnionFind(int N) : par(N), _size(N)
    { // 最初は全てが根であるとして初期化
        for (int i = 0; i < N; i++)
        {
            par[i] = i;
            _size[i] = 1;
        }
    }
    UnionFind() {}

    int root(int x)
    { // データxが属する木の根を再帰で得る：root(x) = {xの木の根}
        if (par[x] == x)
            return x;
        return par[x] = root(par[x]);
    }

    void unite(int x, int y)
    {                     // xとyの木を併合
        int rx = root(x); // xの根をrx
        int ry = root(y); // yの根をry
        if (rx == ry)
            return;   // xとyの根が同じ(=同じ木にある)時はそのまま
        par[rx] = ry; // xとyの根が同じでない(=同じ木にない)時：xの根rxをyの根ryにつける
        _size[ry] += _size[rx];
    }

    bool same(int x, int y)
    { // 2つのデータx, yが属する木が同じならtrueを返す
        int rx = root(x);
        int ry = root(y);
        return rx == ry;
    }
    int size(int x)
    {
        int rx = root(x);
        return _size[rx];
    }
};

// utility functions, structs, etc...
ll getTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long result = tv.tv_sec * 1000LL + tv.tv_usec / 1000LL;
    // cerr << result << endl;
    return result;
}

template <typename T>
void debug(vector<T> v)
{
#ifdef ONLINE_JUDGE
#else
#endif
#ifdef LOCAL_DEBUG
    for (auto vi : v)
    {
        cout << vi << ' ';
    }
    cout << '\n';
#endif
}

void debug(const char *c, const int v)
{
#ifdef ONLINE_JUDGE
#else
#endif
#ifdef LOCAL_DEBUG
    cout << c << v << '\n';
#endif
}

template <typename T>
void debug(set<T> v)
{
#ifdef ONLINE_JUDGE
#else
#endif
#ifdef LOCAL_DEBUG
    for (auto vi : v)
    {
        cout << vi << ' ';
    }
    cout << '\n';
#endif
}

void debug(const char *s)
{
#ifdef ONLINE_JUDGE
#else
#endif
#ifdef LOCAL_DEBUG
    puts(s);
#endif
}

void debug()
{
#ifdef ONLINE_JUDGE
#else
#endif
#ifdef LOCAL_DEBUG
    puts("debug");
#endif
}

tuple<int, int, int> rotate(int dx, int dy, int dz, int axis, int unit)
{
    /* axis 0 (x) and rotate in y-z plane
    [dx,   [[1, 0, 0],
     dy, =  [0, 0, -1]
     dz ]   [0, 1, 0]
    */
    if (axis == 0)
    {
        REP(i, unit % 4)
        {
            swap(dy, dz);
            dy *= -1;
        }
    }
    else if (axis == 1)
    {
        REP(i, unit % 4)
        {
            swap(dx, dz);
            dx *= -1;
        }
    }
    else if (axis == 2)
    {
        REP(i, unit % 4)
        {
            swap(dx, dy);
            dx *= -1;
        }
    }
    return {dx, dy, dz};
}

// 状態
struct STATE
{
    vvi vertex; // vector of vertex of G1
    vector<set<int>> fragment;
    vvvi G;
    int n; // num of V(G_i)
    // vi perm;     // map from V(G_1) to V(G_2)
    vvvi face;   // maintain silhouette of the each graph
    vvvvi count; // store the target silhouette.
    int d;
    vvvvi answer; // maintain the component idx of the graph
    int anscnt;
    UnionFind uf; // first half: vertices of G[0]. second half: vertices of G[1]
    unsigned long long zbhash;
    // map<pair<int, int>, unsigned long long> hashtable;
    STATE(const int _d, const vvvi &vec) : d(_d)
    {
        n = d * d * d;
        face = vec;
        zbhash = 0ULL;
        // initialize array
        count.assign(2, vvvi(2, vvi(d, vi(d, 0))));
        answer.assign(2, vvvi(d, vvi(d, vi(d, 0))));
        vertex.resize(2);
        fragment.resize(2);
        // perm = RandomPermutation(vertex);
        //         debug("----------------");
        // #ifdef DONLINE_JUDGE
        // #else
        //         REP(i, 2)
        //         {
        //             REP(j, 2)
        //             {
        //                 REP(x, d)
        //                 {
        //                     REP(y, d)
        //                     {
        //                         if ((face[i][j][x] >> y) & 1 == 1)
        //                             cout << '1';
        //                         else
        //                             cout << '0';
        //                     }
        //                     cout << '\n';
        //                 }
        //                 debug("----------------");
        //             }
        //         }
        // #endif
        // create two graphs

        REP(gidx, 2)
        {
            debug("Create Vertex");
            REP(v, n)
            if (is_need(v, gidx))
            {
                auto [x, y, z] = ver2coord(v);
                vertex[gidx].push_back(v);
                answer[gidx][x][y][z] = v;
                count[gidx][0][z][x]++;
                count[gidx][1][z][y]++;
                fragment[gidx].insert(v);
            }
            debug(vertex[gidx]);
        }
        // REP(i, 2)
        // {
        //     REP(v, n)
        //     {
        //         REP(u, min(vertex[0].size(), vertex[1].size()))
        //         {
        //             hashtable[{i * n + v, u}] = engine() % (1ULL << 60);
        //         }
        //     }
        // }
        uf = UnionFind(2 * n);

        G.assign(2, vvi(n));
        REP(x, d)
        {
            REP(y, d)
            {
                REP(z, d)
                {
                    int u = coord2ver(x, y, z);
                    if (x < d - 1)
                    {
                        int v = coord2ver(x + 1, y, z);
                        REP(i, 2)
                        {
                            if (is_adjacent(u, v, i))
                            {
                                G[i][u].push_back(v);
                                G[i][v].push_back(u);
                            }
                        }
                    }
                    if (y < d - 1)
                    {
                        int v = coord2ver(x, y + 1, z);
                        REP(i, 2)
                        {
                            if (is_adjacent(u, v, i))
                            {
                                G[i][u].push_back(v);
                                G[i][v].push_back(u);
                            }
                        }
                    }
                    if (z < d - 1)
                    {
                        int v = coord2ver(x, y, z + 1);
                        REP(i, 2)
                        {
                            if (is_adjacent(u, v, i))
                            {
                                G[i][u].push_back(v);
                                G[i][v].push_back(u);
                            }
                        }
                    }
                }
            }
        }
    }
    STATE(){};

    int coord2ver(int x, int y, int z)
    {
        return x * d * d + y * d + z;
    }

    tuple<int, int, int> ver2coord(int u)
    {
        int x, y, z;
        x = u / (d * d);
        u -= x * (d * d);
        y = u / d;
        u -= y * d;
        z = u;
        return {x, y, z};
    }

    bool is_need(int v, int i)
    {
        auto [x, y, z] = ver2coord(v);
        vi coord = {x, y, z};
        // debug(coord);
        if (x >= 0 && x < d && y >= 0 && y < d && z >= 0 && z < d)
            return (((face[i][0][z] >> x) & 1) == 1) && (((face[i][1][z] >> y) & 1) == 1); // in cube and the cell can use to make silhouette.
        else
            return false; // not in cube
    }
    bool is_need(int x, int y, int z, int i)
    {
        if (x >= 0 && x < d && y >= 0 && y < d && z >= 0 && z < d)
            return ((face[i][0][z] >> x) & 1 == 1) && ((face[i][1][z] >> y) & 1 == 1); // in cube and the cell can use to make silhouette.
        else
            return false; // not in cube
    }
    bool is_adjacent(int u, int v, int i)
    {
        auto [ux, uy, uz] = ver2coord(u);
        auto [vx, vy, vz] = ver2coord(v);
        int diff = abs(ux - vx) + abs(uy - vy) + abs(uz - vz);
        // the manhattan distance if exactly one
        bool cond0 = diff == 1;
        // both u, v are candidates for the silhouette f_i
        bool cond1 = ((face[i][0][uz] >> ux) & 1 == 1) && ((face[i][0][vz] >> vx) & 1 == 1);
        // both u, v are candidates for the silhouette r_i
        bool cond2 = ((face[i][1][uz] >> uy) & 1 == 1) && ((face[i][1][vz] >> vy) & 1 == 1);
        bool cond = cond0 && cond1 && cond2;
        return cond;
    }

    void sync()
    {
        map<int, int> m;
        count.assign(2, vvvi(2, vvi(d, vi(d, 0))));
        answer.assign(2, vvvi(d, vvi(d, vi(d, 0))));
        zbhash = 0ULL;
        // int cnt = 0;
        anscnt = 0;
        REP(i, 2)
        {
            for (int v : vertex[i])
            {
                int rx = uf.root(n * i + v);
                int size = uf.size(n * i + v);
                auto [x, y, z] = ver2coord(v);
                bool cond = (size > 1);
                if (m.count(rx) == 0 && cond)
                {
                    anscnt++;
                    m[rx] = anscnt;
                }
                if (cond)
                {
                    answer[i][x][y][z] = m[rx];
                    pair<int, int> key = {i * n + v, m[rx]};
                    if (hashtable.count(key) == 0)
                    {
                        hashtable[key] = engine() % (1ULL << 60);
                    }
                    zbhash ^= hashtable[key];
                    // debug("answer", m[rx]);
                    count[i][0][z][x]++;
                    count[i][1][z][y]++;
                }
            }
        }
        int flagment = 0;
        REP(i, 2)
        {
            int tmp = 0;
            for (int v : vertex[i])
            {
                auto [x, y, z] = ver2coord(v);
                int size = uf.size(n * i + v);
                // reduce
                if (size > 1)
                    continue;
                if (count[i][0][z][x] < 1 || count[i][1][z][y] < 1)
                {
                    // debug("adjust");
                    count[i][0][z][x]++;
                    count[i][1][z][y]++;
                    tmp++;
                    answer[i][x][y][z] = anscnt + tmp;
                    pair<int, int> key = {i * n + v, anscnt + tmp};
                    if (hashtable.count(key) == 0)
                    {
                        hashtable[key] = engine() % (1ULL << 60);
                    }
                    zbhash ^= hashtable[key];
                }
            }
            if (tmp > flagment)
                flagment = tmp;
        }
        anscnt += flagment;
    }

    void output()
    {
        cout << anscnt << '\n';
        REP(i, 2)
        {
            REP(x, d)
            {
                REP(y, d)
                {
                    REP(z, d)
                    {
                        cout << answer[i][x][y][z];
                        if (x == d - 1 && y == d - 1 && z == d - 1)
                            cout << '\n';
                        else
                            cout << ' ';
                    }
                }
            }
        }
    }
};

// 状態のスコア計算
ll calc_score(STATE &state)
{
    // return 0;
    state.sync();
    // state.output();
    vector<int> used(state.anscnt + 1, 0);
    vector<int> size(state.anscnt + 1, 0);
    REP(i, 2)
    {
        for (int v : state.vertex[i])
        {
            auto [x, y, z] = state.ver2coord(v);
            used[state.answer[i][x][y][z]] |= (1 << i);
            size[state.answer[i][x][y][z]]++;
        }
    }
    ll res = 0;
    ll penal = 1000000LL;
    FOR(i, 1, state.anscnt + 1)
    {
        if (used[i] != 3)
        {
            res += penal * size[i];
        }
        res += penal / size[i];
    }
    // debug("score: ", res);
    // debug("hash:", state.zbhash);
    return res;
}

// 状態の初期化
void init(STATE &state)
{
    // puts("Debug");
    int from = 0, to = 1;
    int n = state.n;
    // debug("n=", n);
    vector<long unsigned int> tmp = {state.fragment[from].size(), state.fragment[to].size()};
    debug(tmp);
    if (state.fragment[from].size() > state.fragment[to].size())
    {
        swap(from, to);
    }
    vector<int> sampled;
    sample(all(state.fragment[to]), back_inserter(sampled), state.fragment[to].size(), engine);
    shuffle(all(sampled), engine);
    debug(state.fragment[from]);
    debug(sampled);
    vector<vector<bool>> used(2, vector<bool>(state.n, false));
    int to_i = 0;
    int max_to = state.fragment[to].size();
    // 1. 片方の始点と対応するもう一方の始点をランダムに選択する
    // 2. 回転方向と始点をランダムに選択しBFSして伸ばせるものから伸ばしていく
    // debug("init");
    int cnt = 0;
    for (int r : state.fragment[from])
    {
        // if (cnt > 5)
        //     break;
        int axis = engine() % 3;
        int unit = engine() % 4;

        // G[to] 側でuに対応する頂点を探索
        while (to_i < max_to && used[to][sampled[to_i]])
        {
            to_i++;
        }
        // 全部使い切ってたら(孤立点が残っていなければ)終了
        if (to_i == max_to)
            break;
        int p = sampled[to_i];
        auto [px, py, pz] = state.ver2coord(p);
        // もしrが既にどこかにマージ済なら終了
        if (used[from][r])
            continue;
        auto [rx, ry, rz] = state.ver2coord(r);
        queue<int> que;
        que.push(r);
        while (!que.empty())
        {
            int u = que.front();
            que.pop();
            // auto [ux, uy, uz] = state.ver2coord(u);
            // u から辿れる場所を調べる
            for (int next_u : state.G[from][u])
            {
                auto [next_ux, next_uy, next_uz] = state.ver2coord(next_u);
                // 既に探索済ならスキップ
                if (used[from][next_u])
                    continue;
                // rからのdiffをとる
                int dx = next_ux - rx, dy = next_uy - ry, dz = next_uz - rz;
                // 回転させたdiffをとる
                auto [to_dx, to_dy, to_dz] = rotate(dx, dy, dz, axis, unit);
                // 回転させたdiffをpに作用させてqを得る
                int qx = px + to_dx, qy = py + to_dy, qz = pz + to_dz;
                // qがto側で使われるグラフなら...
                if (state.is_need(qx, qy, qz, to))
                {
                    int q = state.coord2ver(qx, qy, qz);
                    if (used[to][q])
                        continue;
                    state.uf.unite(from * n + next_u, from * n + u);
                    state.uf.unite(to * n + q, to * n + p);
                    que.push(next_u);
                    used[from][u] = true;
                    used[from][next_u] = true;
                    used[to][q] = true;
                }
            }
        }
        if (state.uf.size(from * n + r) > 1)
        {
            state.uf.unite(from * n + r, to * n + p);
            used[from][r] = true;
            used[to][p] = true;
        }
        cnt++;
    }
    // ll score = calc_score(state);
    // debug("score", score);
    // debug("init end");
}

// 状態遷移
void greedy(STATE &state, int r, int p, int from, int to, int axis, int unit)
{
    // int from = 0, to = 1;
    int n = state.n;
    // debug("n=", n);
    // 1. 片方の始点と対応するもう一方の始点をランダムに選択する
    // 2. 回転方向と始点をランダムに選択しBFSして伸ばせるものから伸ばしていく

    auto [px, py, pz] = state.ver2coord(p);
    auto [rx, ry, rz] = state.ver2coord(r);
    queue<int> que;
    que.push(r);
    while (!que.empty())
    {
        int u = que.front();
        que.pop();
        // auto [ux, uy, uz] = state.ver2coord(u);
        // u から辿れる場所を調べる
        for (int next_u : state.G[from][u])
        {
            auto [next_ux, next_uy, next_uz] = state.ver2coord(next_u);
            // 既に探索済ならスキップ
            if (state.fragment[from].find(next_u) == state.fragment[from].end())
                continue;
            // rからのdiffをとる
            int dx = next_ux - rx, dy = next_uy - ry, dz = next_uz - rz;
            // 回転させたdiffをとる
            auto [to_dx, to_dy, to_dz] = rotate(dx, dy, dz, axis, unit);
            // 回転させたdiffをpに作用させてqを得る
            int qx = px + to_dx, qy = py + to_dy, qz = pz + to_dz;
            // qがto側で使われるグラフなら...
            if (state.is_need(qx, qy, qz, to))
            {
                int q = state.coord2ver(qx, qy, qz);
                if (state.fragment[to].find(q) == state.fragment[to].end())
                    continue;
                state.uf.unite(from * n + next_u, from * n + r);
                state.uf.unite(to * n + q, to * n + p);
                que.push(next_u);
                state.fragment[from].erase(next_u);
                state.fragment[to].erase(q);
            }
        }
        // }
        if (state.uf.size(from * n + r) > 1)
        {
            state.uf.unite(from * n + r, to * n + p);
            state.fragment[from].erase(r);
            state.fragment[to].erase(p);
        }
    }
    // ll score = calc_score(state);
    // debug("score", score);
    // debug("greedy end");
}

void greedy(STATE &state, int r, int p, int from, int to, int axis, int unit, int max_depth)
{
    // int from = 0, to = 1;
    int n = state.n;
    // debug("n=", n);
    // 1. 片方の始点と対応するもう一方の始点をランダムに選択する
    // 2. 回転方向と始点をランダムに選択しBFSして伸ばせるものから伸ばしていく

    auto [px, py, pz] = state.ver2coord(p);
    auto [rx, ry, rz] = state.ver2coord(r);
    queue<int> que;
    que.push(r);
    while (!que.empty())
    {
        int u = que.front();
        que.pop();
        // auto [ux, uy, uz] = state.ver2coord(u);
        // u から辿れる場所を調べる
        for (int next_u : state.G[from][u])
        {
            auto [next_ux, next_uy, next_uz] = state.ver2coord(next_u);
            // 既に探索済ならスキップ
            if (state.fragment[from].find(next_u) == state.fragment[from].end())
                continue;
            // rからのdiffをとる
            int dx = next_ux - rx, dy = next_uy - ry, dz = next_uz - rz;
            if (abs(dx) + abs(dy) + abs(dz) > max_depth)
            {
                continue;
            }
            // 回転させたdiffをとる
            auto [to_dx, to_dy, to_dz] = rotate(dx, dy, dz, axis, unit);
            // 回転させたdiffをpに作用させてqを得る
            int qx = px + to_dx, qy = py + to_dy, qz = pz + to_dz;
            // qがto側で使われるグラフなら...
            if (state.is_need(qx, qy, qz, to))
            {
                int q = state.coord2ver(qx, qy, qz);
                if (state.fragment[to].find(q) == state.fragment[to].end())
                    continue;
                state.uf.unite(from * n + next_u, from * n + r);
                state.uf.unite(to * n + q, to * n + p);
                que.push(next_u);
                state.fragment[from].erase(next_u);
                state.fragment[to].erase(q);
            }
        }
        // }
        if (state.uf.size(from * n + r) > 1)
        {
            state.uf.unite(from * n + r, to * n + p);
            state.fragment[from].erase(r);
            state.fragment[to].erase(p);
        }
    }
    // ll score = calc_score(state);
    // debug("score", score);
    // debug("greedy end");
}

void k_best(STATE &state, int k, int start_size, int end_size, ll timelimit)
{
    // STATE best = state;
    // init(state);
    auto compare = [](pair<ll, STATE> a, pair<ll, STATE> b)
    {
        // pair<score, state>
        return a.first > b.first; // minimization
    };
    ll start_time = getTime(); // 開始時刻
    ll score = calc_score(state);
    priority_queue<pair<ll, STATE>, vector<pair<ll, STATE>>, decltype(compare)> que(compare);
    // priority_queue<pair<int, Struct>, vector<pair<ll, STATE>>, compare> que_next;
    que.push({score, state});
    int cnt = 0;
    while (true)
    {
        debug("cnt: ", cnt);
        ll now_time = getTime(); // 現在時刻
        if (now_time - start_time > timelimit)
            break;
        int samplesize = start_size + (end_size - start_size) * (now_time - start_time) / timelimit;
        priority_queue<pair<ll, STATE>, vector<pair<ll, STATE>>, decltype(compare)> que_next(compare);
        // que_next.push(que.top());
        int que_size = que.size();
        debug("que_size: ", que_size);
        set<unsigned long long> hash_set;
        if (que_size == 0)
        {
            return;
        }
        // state = que.top().second;
        REP(i, min(k, que_size)) // 探索幅
        {                        // 時間の許す限り回す
            // ll now_time = getTime(); // 現在時刻
            // if (now_time - start_time > TIME_LIMIT_M)
            //     break;

            auto [score, new_state] = que.top();
            // STATE new_state = que.top().second;
            que.pop();
            // modify_mountain(STATE &state, int r, int p, int from, int to, int axis, int unit)
            vector<vector<int>> sampled(2);
            REP(i, 2)
            {
                // priority_queue<pair<int, int>> deg_que;
                // for (int v : new_state.fragment[i])
                // {
                //     int deg = state.G[i][v].size();
                //     deg_que.push({deg, v});
                // }
                // REP(j, min(k, (int)deg_que.size()))
                // {
                //     int v = deg_que.top().second;
                //     sampled[i].push_back(v);
                //     deg_que.pop();
                // }
                sample(all(new_state.fragment[i]), back_inserter(sampled[i]), samplesize, engine);
                shuffle(all(sampled[i]), engine);
            }

            for (int r : sampled[0])
            {
                for (int p : sampled[1])
                {
                    // priority_queue<pair<ll, STATE>, vector<pair<ll, STATE>>, decltype(compare)> que_tmp(compare);

                    for (int axis = 0; axis < 3; axis++)
                    {
                        for (int unit = 0; unit < 4; unit++)
                        {
                            STATE tmp_state = new_state;
                            greedy(tmp_state, r, p, 0, 1, axis, unit);
                            ll tmp_score = calc_score(tmp_state);
                            if (hash_set.find(tmp_state.zbhash) == hash_set.end())
                            {
                                que_next.push({tmp_score, tmp_state});
                                hash_set.insert(tmp_state.zbhash);
                            }
                        }
                    }
                    // auto [tmp_score, tmp_state] = que_tmp.top();
                    // que_next.push({tmp_score, tmp_state});
                }
            }
        }
        que_size = que_next.size();
        REP(i, min(k, que_size)) // 探索幅
        {
            auto [score, new_state] = que_next.top();
            que.push({score, new_state});
            que_next.pop();
        }
        cnt++;
        // modify_mountain(new_state);
        // debug("modify_end");
        // int new_score = calc_score(new_state);
        // int pre_score = calc_score(state);
        // ll score = calc_score(new_state);
        // if (new_score < pre_score)
        // { // スコア最大化の場合
        //     state = new_state;
        //     // best = new_state;
        // }
        // if (new_score == pre_score)
        // break;
    }
    state = que.top().second;
    // state = best;
}
// 状態遷移
void modify_mountain(STATE &state)
{
    // debug("modify_mountain");
    STATE best_state = state;
    vector<vector<int>> sampled(2);
    REP(i, 2)
    {
        sample(all(state.fragment[i]), back_inserter(sampled[i]), state.fragment[i].size(), engine);
        shuffle(all(sampled[i]), engine);
    }
    // debug(sampled[0]);
    // debug(sampled[1]);
    vector<int> max_itr = {(int)state.fragment[0].size(), (int)state.fragment[1].size()};
    // debug(max_itr);
    vector<int> idx(2, 0);
    REP(i, min(min(max_itr[0], max_itr[1]), 2))
    {
        int from = i % 2, to = (i + 1) % 2;
        while (idx[from] < max_itr[from] && state.fragment[from].find(sampled[from][idx[from]]) == state.fragment[from].end())
        {
            idx[from]++;
            // r = sampled[from][idx[from]];
        }
        if (idx[from] >= max_itr[from])
            break;
        int r = sampled[from][idx[from]];
        while (idx[to] < max_itr[to] && state.fragment[to].find(sampled[to][idx[to]]) == state.fragment[to].end())
        {
            idx[to]++;
            // p = sampled[to][idx[to]];
        }
        if (idx[to] >= max_itr[to])
            break;
        int p = sampled[to][idx[to]];
        // ll pre_score = 0;
        // debug("calc_score: start");
        ll pre_score = calc_score(state);
        // debug("calc_score: done");

        REP(axis, 3)
        {
            REP(unit, 4)
            {
                // debug("copy state");
                STATE new_state = state;
                // debug("axis ", axis);
                // debug("unit ", unit);
                greedy(new_state, r, p, from, to, axis, unit);
                ll new_score = calc_score(new_state);
                if (new_score < pre_score)
                    best_state = new_state;
            }
        }
    }
    state = best_state;
}

void modify_sa(STATE &state, int beamwidth, int samplesize)
{
    int breaksize = min(10, state.anscnt / 2);
    vector<int> ansvec(state.anscnt);
    REP(i, state.anscnt) { ansvec[i] = i; }
    shuffle(all(ansvec), engine);
    set<int> breakid;
    REP(i, min(breaksize, state.anscnt))
    {
        int b = ansvec[i] + 1;
        breakid.insert(b);
    }
    // vvi sampled(2);
    auto compare = [](pair<ll, STATE> a, pair<ll, STATE> b)
    {
        // pair<score, state>
        return a.first > b.first; // minimization
    };
    // priority_queue<pair<ll, STATE>, vector<pair<ll, STATE>>, decltype(compare)> que(compare);
    REP(i, 2)
    {
        for (int b : breakid)
        {
            for (int x = 0; x < state.d; x++)
            {
                for (int y = 0; y < state.d; y++)
                {
                    for (int z = 0; z < state.d; z++)
                    {
                        if (state.answer[i][x][y][z] == b)
                        {
                            int v = state.coord2ver(x, y, z);
                            state.uf.par[state.n * i + v] = state.n * i + v;
                            state.fragment[i].insert(v);
                            // sampled[i].push_back(v);
                        }
                    }
                }
            }
        }
        // sample(all(state.fragment[i]), back_inserter(sampled[i]), samplesize, engine);
        // shuffle(all(sampled[i]), engine);
    }
    k_best(state, beamwidth, samplesize, samplesize, 100);
}

// 山登り法
void mountain(STATE &state)
{
    // STATE best;
    // init(state);

    ll start_time = getTime(); // 開始時刻
    while (true)
    {                            // 時間の許す限り回す
        ll now_time = getTime(); // 現在時刻
        if (now_time - start_time > TIME_LIMIT_M)
            break;

        STATE new_state = state;

        // modify_mountain(STATE &state, int r, int p, int from, int to, int axis, int unit)
        modify_mountain(new_state);
        // debug("modify_end");
        ll new_score = calc_score(new_state);
        ll pre_score = calc_score(state);

        if (new_score < pre_score)
        { // スコア最大化の場合
            state = new_state;
            // best = new_state;
        }
        // if (new_score == pre_score)
        // break;
    }
}

// 焼きなまし法
void sa(STATE &state, int beamwidth, int samplesize, ll time_limit)
{
    STATE best = state;
    int score = calc_score(state);
    // init(state);
    // mountain(state);
    ll start_temp = score, end_temp = 0; // 適当な値を入れる（後述）
    ll start_time = getTime();           // 開始時刻
    while (true)
    {                            // 時間の許す限り回す
        ll now_time = getTime(); // 現在時刻
        if (now_time - start_time > time_limit)
            break;

        STATE new_state = state;
        modify_sa(new_state, beamwidth, samplesize);
        int new_score = calc_score(new_state);
        int pre_score = calc_score(state);

        // 温度関数
        double temp = (double)start_temp + (end_temp - start_temp) * (now_time - start_time) / (double)time_limit;
        // 遷移確率関数(最大化の場合)
        double prob = exp((pre_score - new_score) / temp);
        debug("pre_score: ", pre_score);
        debug("new_score: ", new_score);
        debug("temp: ", temp);
        debug("prob: ", prob);

        if (prob > (rand() % INF) / (double)INF)
        { // 確率probで遷移する
            state = new_state;
        }
        if (new_score < pre_score)
        {
            best = new_state;
        }
    }
    state = best;
}

void solve(const int d, const vvvi &face)
{
    STATE state(d, face);
    // modify_mountain(state);
    // mountain(state);
    // int samplesize = 5;
    int beamwidth = 1;
    int samplesize = round((1. - ((double)d - 5.) / 9.) * 5. + ((double)d - 5.) / 9. * 2.);
    // int beamwidth = round((1. - ((double)d - 5.) / 9.) * 15. + ((double)d - 5.) / 9. * 3.);
    debug("start k_best");
    k_best(state, beamwidth, samplesize, samplesize, TIME_LIMIT);
    debug("end k_best");
    // debug("score: ", calc_score(state));
    // debug("start sa");
    // sa(state, beamwidth, samplesize, TIME_LIMIT_SA);
    // debug("end sa");
    // debug("score: ", calc_score(state));
    state.sync();
    state.output();
}

int main()
{
    // face[i][j]: f_i (if j = 0) or r_i (if j = 1)
    // f_i = [l1, l2, l3...] l = 01110001
    vvvi face;
    int d;

    cin >> d;
    string s;
    face.assign(2, vvi(2, vi(d, 0)));
    // cout << d << endl;
    REP(i, 2)
    {
        REP(j, 2)
        {
            REP(k, d)
            {
                cin >> s;
                REP(l, d)
                if (s[l] == '1')
                {
                    face[i][j][k] += (1 << l);
                }
            }
        }
    }
    debug();

    solve(d, face);
}