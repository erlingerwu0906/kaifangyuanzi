#include <stdio.h>
#include <stdlib.h>

int *parent;
int *rank;
int count;

void init(int n) {
    parent = (int *)malloc((n + 1) * sizeof(int));
    rank = (int *)malloc((n + 1) * sizeof(int));
    count = n;
    for (int i = 1; i <= n; i++) {
        parent[i] = i;
        rank[i] = 0;
    }
}

int find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}

void unite(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    if (rootX == rootY) return;
    if (rank[rootX] < rank[rootY]) {
        parent[rootX] = rootY;
    } else if (rank[rootX] > rank[rootY]) {
        parent[rootY] = rootX;
    } else {
        parent[rootY] = rootX;
        rank[rootX]++;
    }
    count--;
}

int isAllConnected() {
    return count == 1;
}

int main() {
    int N, M;
    scanf("%d %d", &N, &M);
    init(N);
    int result = -1;
    for (int day = 1; day <= M; day++) {
        int a, b;
        scanf("%d %d", &a, &b);
        int rootA = find(a);
        int rootB = find(b);
        if (rootA != rootB) {
            unite(a, b);
        }
        if (isAllConnected()) {
            result = day;
            break;
        }
    }
    printf("%d\n", result);
    free(parent);
    free(rank);
    return 0;
}