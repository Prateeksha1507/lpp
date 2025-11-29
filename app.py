from flask import Flask, request, jsonify, render_template
import math

app = Flask(__name__)

class BranchAndBoundILP:
    def __init__(self, adj, source, target):
        self.adj = adj
        self.n = len(adj)
        self.source = source
        self.target = target
        self.edges = []
        self.best_cost = math.inf
        self.best_solution = None

        for i in range(self.n):
            for j in range(self.n):
                # treat 0 or None as no edge
                if adj[i][j] not in (0, None):
                    self.edges.append((i, j, adj[i][j]))

    def check_constraints(self, solution):
        
        flow = [0]*self.n
        cost = 0

        for i in range(len(self.edges)):
            if solution[i] == 1:
                u, v, w = self.edges[i]
                flow[u] += 1
                flow[v] -= 1
                cost += w

        for i in range(self.n):
            if i == self.source and flow[i] != 1:
                return False, None
            elif i == self.target and flow[i] != -1:
                return False, None
            elif i not in (self.source,self.target) and flow[i] != 0:
                return False, None

        return True, cost

    def compute_lower_bound(self, solution):
        
        return sum(self.edges[i][2] for i in range(len(solution)) if solution[i] == 1)

    def branch_and_bound(self, solution, idx):
        
        bound = self.compute_lower_bound(solution)
        if bound >= self.best_cost:
            return

        
        if idx == len(solution):
            valid, cost = self.check_constraints(solution)
            if valid and cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = solution.copy()
            return

        
        solution[idx] = 1
        self.branch_and_bound(solution, idx+1)

        
        solution[idx] = 0
        self.branch_and_bound(solution, idx+1)

    def solve(self):
       
        if self.source < 0 or self.source >= self.n or self.target < 0 or self.target >= self.n:
            return None, None

        
        if not self.edges:
            return None, None

        solution = [0] * len(self.edges)
        self.branch_and_bound(solution, 0)

        if self.best_solution is None:
            return None, None

        path = [self.source]
        curr = self.source
        steps = 0
        max_steps = self.n + 5
        while curr != self.target and steps < max_steps:
            steps += 1
            progressed = False
            for i, use in enumerate(self.best_solution):
                if use == 1:
                    u, v, _ = self.edges[i]
                    if u == curr and v not in path:
                        path.append(v)
                        curr = v
                        progressed = True
                        break
            if not progressed:
                return None, None

        if curr != self.target:
            return None, None

        return path, self.best_cost

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/solve", methods=["POST"])
def solve_route():
    """
    Expects JSON:
    {
      "adj": [[...], [...], ...],
      "start": 0,
      "end": 3]
    }
    """
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        adj = data["adj"]
        start = int(data["start"])
        end = int(data["end"])
    except Exception as e:
        return jsonify({"error": "Missing/invalid fields", "detail": str(e)}), 400

    n = len(adj)
    if any(len(row) != n for row in adj):
        return jsonify({"error": "Adjacency matrix must be square"}), 400

    solver = BranchAndBoundILP(adj, start, end)
    path, cost = solver.solve()

    if path is None:
        return jsonify({"path": None, "cost": None, "message": "No path found"}), 200

    return jsonify({"path": path, "cost": cost}), 200

if __name__ == "__main__":
    
    app.run(debug=True)
