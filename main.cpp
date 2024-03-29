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
constexpr long long TIME_LIMIT = 5500;
constexpr long long TIME_LIMIT_M = 3000;
constexpr long long TIME_LIMIT_SA = 2000;
constexpr long long INF = 1000000000LL;
ll penal = 1000000LL;

// global
map<pair<int, int>, unsigned long long> hashtable; // (v, rv) -> hash;
vvvi face;                                         // maintain silhouette of the each graph
vvvi G;
vvi vertex; // vector of vertex of G1

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
        if (ry > rx)
            swap(rx, ry);
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

void testinfo(const char *c, const int v)
{
#ifdef LOCAL_TEST
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
    vector<set<int>> fragment;
    // vvvi G;
    int n; // num of V(G_i)
    // vi perm;     // map from V(G_1) to V(G_2)
    vvvvi count; // store the target silhouette. count[i][j][z][x or y]
    // vvvvi count_flg; // store the target silhouette. count[i][j][z][x or y]
    int d;
    // vvvvi answer; // maintain the component idx of the graph
    int anscnt;
    ll merged_score;
    ll flagment_score;
    UnionFind uf; // first half: vertices of G[0]. second half: vertices of G[1]
    unsigned long long zbhash;
    ll score;
    // map<pair<int, int>, unsigned long long> hashtable;
    STATE(const int _d) : d(_d)
    {
        n = d * d * d;
        // face = vec;
        zbhash = 0ULL;
        // initialize array
        count.assign(2, vvvi(2, vvi(d, vi(d, 0))));
        // count_flg.assign(2, vvvi(2, vvi(d, vi(d, 0))));
        // answer.assign(2, vvvi(d, vvi(d, vi(d, 0))));
        vertex.resize(2);
        fragment.resize(2);
        merged_score = 0LL;
        flagment_score = 0LL;
        // score = n * 1000000LL;
        score = 0LL;
        // n * 1000000LL;
        REP(gidx, 2)
        {
            debug("Create Vertex");
            REP(v, n)
            if (is_need(v, gidx))
            {
                auto [x, y, z] = ver2coord(v);
                vertex[gidx].push_back(v);
                // answer[gidx][x][y][z] = v;
                // count[gidx][0][z][x]++;
                // count[gidx][1][z][y]++;
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

    void unite(int u, int v, int ui, int vi)
    {
        auto [ux, uy, uz] = ver2coord(u);
        auto [vx, vy, vz] = ver2coord(v);
        // int usize = uf.size(ui * n + u);
        // int vsize = uf.size(vi * n + v);
        uf.unite(ui * n + u, vi * n + v);
        if (uf.size(vi * n + v) > 1)
        {
            merged_score += penal / (uf.size(vi * n + v) / 2);
        }
        count[ui][0][uz][ux]++;
        count[ui][1][uz][uy]++;
        count[vi][0][vz][vx]++;
        count[vi][1][vz][vy]++;
    }
    void unite(int u, int v, int i)
    {
        auto [ux, uy, uz] = ver2coord(u);
        auto [vx, vy, vz] = ver2coord(v);
        int usize = uf.size(i * n + u);
        int vsize = uf.size(i * n + v);
        uf.unite(i * n + u, i * n + v);
        // if (uf.size(i * n + v) > 1)
        // {
        //     merged_score -= 1000000LL / uf.size(i * n + v);
        // }
        count[i][0][uz][ux]++;
        count[i][1][uz][uy]++;
        // count[i][0][vz][vx]++;
        // count[i][1][vz][vy]++;
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

    void sync_light()
    {
        ll flg = 0;
        vvvvi count_flg; // store the target silhouette. count[i][j][z][x or y]
        count_flg.assign(2, vvvi(2, vvi(d, vi(d, 0))));
        REP(i, 2)
        {
            ll tmp = 0;
            for (int v : fragment[i])
            {
                auto [x, y, z] = ver2coord(v);
                if (count_flg[i][0][z][x] + count[i][0][z][x] < 1 || count_flg[i][1][z][y] + count[i][1][z][y] < 1)
                {
                    count_flg[i][0][z][x]++;
                    count_flg[i][1][z][y]++;
                    tmp++;
                }
            }
            if (flg < tmp)
                flg = tmp;
        }
        flagment_score = penal * flg;
        score = merged_score + flagment_score;
    }

    void calc_score()
    {
        sync_light();
    }

    void update_hash(vector<vector<int>> &dirty_vertex)
    {
        REP(i, 2)
        {
            for (int v : dirty_vertex[i])
            {
                int rx = uf.root(n * i + v);
                pair<int, int> key = {i * n + v, rx};
                if (hashtable.count(key) == 0)
                {
                    hashtable[key] = engine() % (1ULL << 60);
                }
                zbhash ^= hashtable[key];
            }
        }
    }
};

void output(STATE &state)
{
    map<int, int> m;
    int d = state.d;
    int n = state.n;
    vvvvi count;
    vvvvi answer;
    count.assign(2, vvvi(2, vvi(d, vi(d, 0))));
    answer.assign(2, vvvi(d, vvi(d, vi(d, 0))));
    // int cnt = 0;
    int anscnt = 0;
    REP(i, 2)
    {
        for (int v : vertex[i])
        {
            int rx = state.uf.root(n * i + v);
            int size = state.uf.size(n * i + v);
            auto [x, y, z] = state.ver2coord(v);
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
    int flg = 0;
    REP(i, 2)
    {
        int tmp = 0;
        for (int v : vertex[i])
        {
            auto [x, y, z] = state.ver2coord(v);
            int size = state.uf.size(n * i + v);
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
            }
        }
        if (tmp > flg)
            flg = tmp;
    }
    anscnt += flg;

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

// 状態のスコア計算
ll calc_score(STATE &state)
{
    state.calc_score();
    // debug("score: ", state.score);
    return state.score;
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
    queue<pair<int, int>> que;
    que.push({r, 0});
    vvi dirty_vertex(2); // hash計算する対象
    while (!que.empty())
    {
        auto [u, depth] = que.front();
        que.pop();
        if (depth > max_depth)
        {
            continue;
        }
        // auto [ux, uy, uz] = state.ver2coord(u);
        // u から辿れる場所を調べる
        depth++;
        for (int next_u : G[from][u])
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
                state.unite(next_u, r, from);
                state.unite(q, p, to);
                que.push({next_u, depth});
                dirty_vertex[from].push_back(next_u);
                dirty_vertex[to].push_back(q);
                state.fragment[from].erase(next_u);
                state.fragment[to].erase(q);
            }
        }
    }
    if (state.uf.size(from * n + r) > 1)
    {
        // state.uf.unite(from * n + r, to * n + p);
        state.unite(r, p, from, to);
        state.fragment[from].erase(r);
        state.fragment[to].erase(p);
        dirty_vertex[from].push_back(r);
        dirty_vertex[to].push_back(p);
        state.update_hash(dirty_vertex);
    }
}

// // 状態遷移
// void greedy(STATE &state, int r, int p, int from, int to, int axis, int unit)
// {
//     greedy(state, r, p, from, to, axis, unit, 10);
// }

void k_best(STATE &state, int k, int start_size, int end_size, int maxdepth, int threthold, ll timelimit)
{
    auto compare = [](pair<ll, STATE> a, pair<ll, STATE> b)
    {
        // pair<score, state>
        return a.first > b.first; // minimization
    };
    ll start_time = getTime(); // 開始時刻
    ll score = calc_score(state);
    priority_queue<pair<ll, STATE>, vector<pair<ll, STATE>>, decltype(compare)> que(compare);
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
        int que_size = que.size();
        debug("que_size: ", que_size);
        set<unsigned long long> hash_set;
        set<ll> score_set;
        if (que_size == 0)
        {
            return;
        }
        REP(i, min(k, que_size)) // 探索幅
        {                        // 時間の許す限り回す
            auto [score, new_state] = que.top();
            que.pop();
            vector<vector<int>> sampled(2);
            REP(i, 2)
            {
                sample(all(new_state.fragment[i]), back_inserter(sampled[i]), samplesize, engine);
                shuffle(all(sampled[i]), engine);
            }

            for (int r : sampled[0])
            {
                for (int p : sampled[1])
                {
                    if (state.d > threthold)
                    {
                        int axis = engine() % 3;
                        int unit = engine() % 4;
                        STATE tmp_state = new_state;
                        greedy(tmp_state, r, p, 0, 1, axis, unit, maxdepth);
                        ll tmp_score = calc_score(tmp_state);
                        if (hash_set.find(tmp_state.zbhash) == hash_set.end())
                        {
                            que_next.push({tmp_score, tmp_state});
                            hash_set.insert(tmp_state.zbhash);
                            debug("score: ", tmp_score);
                        }
                    }
                    else
                    {

                        for (int axis = 0; axis < 3; axis++)
                        {
                            for (int unit = 0; unit < 4; unit++)
                            {
                                STATE tmp_state = new_state;
                                greedy(tmp_state, r, p, 0, 1, axis, unit, maxdepth);
                                ll tmp_score = calc_score(tmp_state);
                                if (hash_set.find(tmp_state.zbhash) == hash_set.end())
                                {
                                    que_next.push({tmp_score, tmp_state});
                                    hash_set.insert(tmp_state.zbhash);
                                    debug("score: ", tmp_score);
                                }
                            }
                        }
                    }
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
    }
    state = que.top().second;
    // state = best;
}

void solve(const int d, const vvvi &face,
           pair<double, double> r_s_samplesize,
           pair<double, double> r_e_samplesize,
           pair<double, double> r_beamwidth,
           pair<double, double> r_maxdepth,
           int threshold)
{
    STATE state(d);
    int s_samplesize = round((1. - ((double)d - 5.) / 9.) * r_s_samplesize.first + ((double)d - 5.) / 9. * r_s_samplesize.second); // 10 - 5
    int e_samplesize = round((1. - ((double)d - 5.) / 9.) * r_e_samplesize.first + ((double)d - 5.) / 9. * r_e_samplesize.second); // 10 - 5
    int beamwidth = round((1. - ((double)d - 5.) / 9.) * r_beamwidth.first + ((double)d - 5.) / 9. * r_beamwidth.second);          // 10 - 5
    int maxdepth = round((1. - ((double)d - 5.) / 9.) * r_maxdepth.first + ((double)d - 5.) / 9. * r_maxdepth.second);             // 10 - 7
    if (d <= 6)
    {
        s_samplesize = 10;
        e_samplesize = 10;
        beamwidth = 10;
        maxdepth = 10;
    }
    debug("start k_best");
    k_best(state, beamwidth, s_samplesize, e_samplesize, maxdepth, threshold, TIME_LIMIT);
    ll score = state.score;
    debug("end k_best");
    output(state);
    testinfo("score: ", score);
}

int main(int argc, char *argv[])
{
#ifdef LOCAL_TEST
    if (argc != 10)
    {
        cout << "Invalid arguments number: expected 9, but " << argc << " arguments given\n";
        return 9;
    }
    pair<double, double> s_samplesize = {stod(argv[1]), stod(argv[2])};
    pair<double, double> e_samplesize = {stod(argv[3]), stod(argv[4])};
    pair<double, double> beamwidth = {stod(argv[5]), stod(argv[6])};
    pair<double, double> maxdepth = {stod(argv[7]), stod(argv[8])};
    int threshold = stod(argv[9]);
#else
    // face[i][j]: f_i (if j = 0) or r_i (if j = 1)
    // f_i = [l1, l2, l3...] l = 01110001
    // vvvi face;
    pair<double, double> s_samplesize = {3., 12.};
    pair<double, double> e_samplesize = {14., 2.};
    pair<double, double> beamwidth = {17., 17.};
    pair<double, double> maxdepth = {5., 19.};
    int threshold = 8;
#endif
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
    // debug();
    solve(d, face, s_samplesize, e_samplesize, beamwidth, maxdepth, threshold);
}
