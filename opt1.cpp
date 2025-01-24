#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <sstream>

using namespace std;

class Graph {
public:
    int V; // Количество вершин
    vector<vector<int>> adj; // Список смежности

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v); // Неориентированный граф
    }

    void greedyColoring() {
        vector<int> result(V, -1); // Хранит назначенные цвета
        vector<bool> available(V, false); // Отслеживает доступные цвета

        result[0] = 0; // Назначаем первый цвет первой вершине

        for (int u = 1; u < V; u++) {
            // Помечаем все цвета как доступные
            fill(available.begin(), available.end(), false);

            // Проверяем цвета соседних вершин
            for (int i : adj[u]) {
                if (result[i] != -1) {
                    available[result[i]] = true;
                }
            }

            // Находим первый доступный цвет
            int cr;
            for (cr = 0; cr < V; cr++) {
                if (!available[cr]) break;
            }
            result[u] = cr; // Назначаем найденный цвет вершине u
        }

        // Подсчитываем цвета и подготавливаем вывод
        set<int> uniqueColors(result.begin(), result.end());
        cout << uniqueColors.size() << endl; // Количество цветов

        vector<vector<int>> colorClasses(uniqueColors.size());
        for (int i = 0; i < V; i++) {
            colorClasses[result[i]].push_back(i + 1); // Храним индексы с 1
        }

        for (const auto& classList : colorClasses) {
            if (!classList.empty()) {
                cout << "{";
                for (size_t j = 0; j < classList.size(); j++) {
                    cout << classList[j];
                    if (j < classList.size() - 1) cout << ",";
                }
                cout << "}" << endl;
            }
        }
    }
};

void readGraphFromFile(const string& filename, Graph& g) {
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        if (line[0] == 'p') { // Строка с проблемой
            istringstream iss(line);
            string type;
            int vertices, edges;
            iss >> type >> type >> vertices >> edges;
            g = Graph(vertices);
        } else if (line[0] == 'e') { // Строка с ребром
            int v1, v2;
            istringstream iss(line);
            iss >> v1 >> v2;
            g.addEdge(v1 - 1, v2 - 1); // Преобразуем в нумерацию с нуля
        }
    }
}

int main() {
    string filename = "myciel3.col"; // Измените на ваш файл входных данных
    Graph g(0);
    
    readGraphFromFile(filename, g);
    g.greedyColoring();

    return 0;
}
