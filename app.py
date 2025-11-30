from flask import Flask, request, jsonify, render_template
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, LpStatus

app = Flask(__name__)

class ILPShortestPath:
    def __init__(self, adj, source, target):
        self.adj = adj
        self.n = len(adj)
        self.source = source
        self.target = target

    def solve(self):
        n = self.n
        prob = LpProblem("ShortestPathILP", LpMinimize)

        # Create binary decision variables for existing edges
        x = {}
        for i in range(n):
            for j in range(n):
                if self.adj[i][j] not in (0, None):
                    x[(i,j)] = LpVariable(f"x_{i}_{j}", cat=LpBinary)

        # Objective: minimize total cost
        prob += lpSum(self.adj[i][j] * x[(i,j)] for (i,j) in x)

        # Flow conservation constraints
        for i in range(n):
            inflow = lpSum(x[(j,i)] for (j,k) in x if k==i)
            outflow = lpSum(x[(i,j)] for (i2,j) in x if i2==i)
            if i == self.source:
                prob += (outflow - inflow == 1)
            elif i == self.target:
                prob += (inflow - outflow == 1)
            else:
                prob += (inflow - outflow == 0)

        # Solve the ILP
        prob.solve()

        if LpStatus[prob.status] != "Optimal":
            return None, None

        # Reconstruct path from selected edges
        path = [self.source]
        curr = self.source
        visited = set()
        while curr != self.target:
            visited.add(curr)
            progressed = False
            for (i,j), var in x.items():
                if i == curr and var.varValue > 0.5 and j not in visited:
                    path.append(j)
                    curr = j
                    progressed = True
                    break
            if not progressed:
                return None, None  # no valid path

        # Compute total cost
        cost = sum(self.adj[i][j] for i,j in zip(path[:-1], path[1:]))
        return path, cost


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
      "end": 3
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

    solver = ILPShortestPath(adj, start, end)
    path, cost = solver.solve()

    if path is None:
        return jsonify({"path": None, "cost": None, "message": "No path found"}), 200

    return jsonify({"path": path, "cost": cost}), 200


if __name__ == "__main__":
    app.run(debug=True)
