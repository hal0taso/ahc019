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
const long long TIME_LIMIT = 5000;
const long long TIME_LIMIT_M = 5000;
constexpr long long INF = 1000000000LL;

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

vi RandomPermutation(const vi &v)
{
    // int n = v.size();
    // int part = 2;
    // int rem = n % part;
    // vector<int> w(v.size() / part);
    // REP(i, w.size())
    // {
    //     w[i] = i;
    // }
    // shuffle(w.begin(), w.end(), engine);
    // vi ans(n);
    // REP(i, w)
    // {
    //     REP(j, part)
    //     {
    //         ans[i * part + j] = w[i] * part + j;
    //     }
    // }
    return v;
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
    vector<set<int>> flagment;
    vvvi G;
    int n; // num of V(G_i)
    // vi perm;     // map from V(G_1) to V(G_2)
    vvvi face;   // maintain silhouette of the each graph
    vvvvi count; // store the target silhouette.
    int d;
    vvvvi answer; // maintain the component idx of the graph
    int anscnt;
    UnionFind uf; // first half: vertices of G[0]. second half: vertices of G[1]

    STATE(const int _d, const vvvi &vec) : d(_d)
    {
        n = d * d * d;
        face = vec;

        // initialize array
        count.assign(2, vvvi(2, vvi(d, vi(d, 0))));
        answer.assign(2, vvvi(d, vvi(d, vi(d, 0))));
        vertex.resize(2);
        flagment.resize(2);
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
                flagment[gidx].insert(v);
            }
            debug(vertex[gidx]);
        }
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
                    // debug("answer", m[rx]);
                    count[i][0][z][x]++;
                    count[i][1][z][y]++;
                }
            }
        }
        REP(i, 2)
        {
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
                    anscnt++;
                    answer[i][x][y][z] = anscnt;
                }
            }
        }
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

// 状態の初期化
void init(STATE &state)
{
    // puts("Debug");
    int from = 0, to = 1;
    int n = state.n;
    debug("n=", n);
    vector<long unsigned int> tmp = {state.flagment[from].size(), state.flagment[to].size()};
    debug(tmp);
    if (state.flagment[from].size() > state.flagment[to].size())
    {
        swap(from, to);
    }
    vector<int> sampled;
    sample(all(state.flagment[to]), back_inserter(sampled), state.flagment[to].size(), engine);
    shuffle(all(sampled), engine);
    debug(state.flagment[from]);
    debug(sampled);
    vector<vector<bool>> used(2, vector<bool>(state.n, false));
    int to_i = 0;
    int max_to = state.flagment[to].size();
    // 1. 片方の始点と対応するもう一方の始点をランダムに選択する
    // 2. 回転方向と始点をランダムに選択しBFSして伸ばせるものから伸ばしていく
    debug("init");
    int cnt = 0;
    for (int r : state.flagment[from])
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
    debug("init end");
}

// 状態遷移
void modify_mountain(STATE &state)
{
}
// 状態遷移
// void modify_mountain(STATE &state, int r, int from, int to, int axis, int unit)
// {
//     // puts("Debug");
//     // int from = 0, to = 1;
//     int n = state.n;
//     debug("n=", n);
//     // vector<long unsigned int> tmp = {state.flagment[from].size(), state.flagment[to].size()};
//     // debug(tmp);
//     // if (state.flagment[from].size() > state.flagment[to].size())
//     // {
//     //     swap(from, to);
//     // }
//     vector<int> sampled;
//     sample(all(state.flagment[to]), back_inserter(sampled), state.flagment[to].size(), engine);
//     shuffle(all(sampled), engine);
//     debug(state.flagment[from]);
//     debug(sampled);
//     vector<vector<bool>> used(2, vector<bool>(state.n, false));
//     int to_i = 0;
//     int max_to = state.flagment[to].size();
//     // 1. 片方の始点と対応するもう一方の始点をランダムに選択する
//     // 2. 回転方向と始点をランダムに選択しBFSして伸ばせるものから伸ばしていく
//     debug("init");
//     int cnt = 0;
//     // for (int r : state.flagment[from])
//     // {
//     // if (cnt > 5)
//     //     break;
//     // int axis = engine() % 3;
//     // int unit = engine() % 4;

//     // G[to] 側でuに対応する頂点を探索
//     while (to_i < max_to && used[to][sampled[to_i]])
//     {
//         to_i++;
//     }
//     // 全部使い切ってたら(孤立点が残っていなければ)終了
//     if (to_i == max_to)
//         break;
//     int p = sampled[to_i];
//     auto [px, py, pz] = state.ver2coord(p);
//     // もしrが既にどこかにマージ済なら終了
//     if (used[from][r])
//         continue;
//     auto [rx, ry, rz] = state.ver2coord(r);
//     queue<int> que;
//     que.push(r);
//     while (!que.empty())
//     {
//         int u = que.front();
//         que.pop();
//         // auto [ux, uy, uz] = state.ver2coord(u);
//         // u から辿れる場所を調べる
//         for (int next_u : state.G[from][u])
//         {
//             auto [next_ux, next_uy, next_uz] = state.ver2coord(next_u);
//             // 既に探索済ならスキップ
//             if (state.flagment[from].find(next_u) == state.flagment[from].end())
//                 continue;
//             // rからのdiffをとる
//             int dx = next_ux - rx, dy = next_uy - ry, dz = next_uz - rz;
//             // 回転させたdiffをとる
//             auto [to_dx, to_dy, to_dz] = rotate(dx, dy, dz, axis, unit);
//             // 回転させたdiffをpに作用させてqを得る
//             int qx = px + to_dx, qy = py + to_dy, qz = pz + to_dz;
//             // qがto側で使われるグラフなら...
//             if (state.is_need(qx, qy, qz, to))
//             {
//                 int q = state.coord2ver(qx, qy, qz);
//                 if (state.flagment[to].find(q) == state.flagment[to].end())
//                     continue;
//                 state.uf.unite(from * n + next_u, from * n + u);
//                 state.uf.unite(to * n + q, to * n + p);
//                 que.push(next_u);
//                 // used[from][u] = true;
//                 // used[from][next_u] = true;
//                 // used[to][q] = true;
//                 state.fragment[from].erase(next_u);
//                 state.fragment[from].erase(u);
//                 state.fragment[from].erase(r);
//                 state.fragment[to].erase(q);
//                 state.fragment[to].erase(p);
//             }
//         }
//         // }
//         if (state.uf.size(from * n + r) > 1)
//         {
//             state.uf.unite(from * n + r, to * n + p);
//             // used[from][r] = true;
//             // used[to][p] = true;
//             state.fragment[from].erase(r);
//             state.fragment[from].erase(q);
//         }
//         cnt++;
//     }
//     debug("init end");
// }

void modify_sa(STATE &state)
{
}

// 状態のスコア計算
int calc_score(STATE &state)
{
    state.sync();
    vector<int> used(state.anscnt, 0);
    vector<int> size(state.anscnt, 0);
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
    ll penal = 1000000000LL;
    REP(i, state.anscnt)
    {
        if (used[i] != 3)
        {
            res += penal * size[i];
        }
        res += penal / size[i];
    }
    return res;
}

// 山登り法
void mountain(STATE &state)
{
    // STATE state;
    // init(state);
    ll start_time; // 開始時刻
    while (true)
    {                // 時間の許す限り回す
        ll now_time; // 現在時刻
        if (now_time - start_time > TIME_LIMIT_M)
            break;

        STATE new_state = state;
        modify_mountain(new_state);
        int new_score = calc_score(new_state);
        int pre_score = calc_score(state);

        if (new_score > pre_score)
        { // スコア最大化の場合
            state = new_state;
        }
    }
}

// 焼きなまし法
void sa(STATE &state)
{
    STATE best;
    // init(state);
    // mountain(state);
    ll start_temp, end_temp; // 適当な値を入れる（後述）
    ll start_time;           // 開始時刻
    while (true)
    {                // 時間の許す限り回す
        ll now_time; // 現在時刻
        if (now_time - start_time > TIME_LIMIT)
            break;

        STATE new_state = state;
        modify_sa(new_state);
        int new_score = calc_score(new_state);
        int pre_score = calc_score(state);

        // 温度関数
        double temp = (double)start_temp + (end_temp - start_temp) * (now_time - start_time) / (double)TIME_LIMIT;
        // 遷移確率関数(最大化の場合)
        double prob = exp((new_score - pre_score) / temp);

        if (prob > (rand() % INF) / (double)INF)
        { // 確率probで遷移する
            state = new_state;
        }
        if (new_score > pre_score)
        {
            best = new_state;
        }
    }
}

void solve(const int d, const vvvi &face)
{
    STATE state(d, face);
    init(state);
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