from flask import Flask, request, jsonify, render_template
import math
import copy
import pulp as lp

app = Flask(__name__)

import math
import copy
import pulp as lp


def is_integer_assignment(x_vars, tol=1e-5):
    """
    Check if all LP variables are (approximately) integer.
    x_vars: dict[edge] -> pulp variable
    """
    for var in x_vars.values():
        val = var.value()
        if val is None:
            return False
        if abs(val - round(val)) > tol:
            return False
    return True


def pick_fractional_variable(x_vars, tol=1e-5):
    """
    Pick one variable that is fractional in the current LP solution.
    Returns (edge_key, value) or (None, None) if none found.
    """
    for key, var in x_vars.items():
        val = var.value()
        if val is None:
            continue
        if abs(val - round(val)) > tol:
            return key, val
    return None, None


class BranchAndBoundILP:
    def __init__(self, adj, source, target):
        """
        adj: adjacency matrix; adj[i][j] = cost or 0/None if no edge
        source: start node index
        target: end node index
        """
        self.adj = adj
        self.n = len(adj)
        self.source = source
        self.target = target

        self.edges = []       
        self.edge_cost = {}    

        for i in range(self.n):
            for j in range(self.n):
                if adj[i][j] not in (0, None):
                    self.edges.append((i, j))
                    self.edge_cost[(i, j)] = float(adj[i][j])

        self.best_cost = math.inf
        self.best_solution = None  #

    def build_lp_relaxation(self):
        """
        Build LP relaxation of the binary shortest-path ILP:
          - variables x_(u,v) ∈ [0,1], continuous
          - objective: minimize sum(cost * x)
          - flow constraints from source to target
        Returns (lp_model, x_vars)
        """
        model = lp.LpProblem("ShortestPathLP", lp.LpMinimize)

        
        x_vars = {}
        for (u, v) in self.edges:
            name = f"x_{u}_{v}"
            x_vars[(u, v)] = lp.LpVariable(name, lowBound=0, upBound=1, cat="Continuous")

        
        obj_expr = 0
        for e in self.edges:
            obj_expr += self.edge_cost[e] * x_vars[e]
        model += obj_expr

        for node in range(self.n):
            out_edges = [e for e in self.edges if e[0] == node]
            in_edges  = [e for e in self.edges if e[1] == node]

            out_sum = lp.lpSum(x_vars[e] for e in out_edges)
            in_sum  = lp.lpSum(x_vars[e] for e in in_edges)

            if node == self.source:
                model += (out_sum - in_sum == 1)
            elif node == self.target:
                model += (out_sum - in_sum == -1)
            else:
                model += (out_sum - in_sum == 0)

        return model, x_vars

    def branch_and_bound(self, model, x_vars, depth=0):
        """
        Manual Branch and Bound on top of LP relaxation.
        model: current LP model
        x_vars: dict[(u,v)] -> pulp variable in this model
        """
        
        model.solve(lp.PULP_CBC_CMD(msg=False))
        status = lp.LpStatus[model.status]

        if status != "Optimal":
            return  

        lp_value = lp.value(model.objective)

        if lp_value >= self.best_cost - 1e-8:
            return

        if is_integer_assignment(x_vars):
            solution = {}
            for e, var in x_vars.items():
                solution[e] = int(round(var.value()))
            cost = sum(self.edge_cost[e] for e, val in solution.items() if val == 1)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = solution
            return

        edge_to_branch, val = pick_fractional_variable(x_vars)
        if edge_to_branch is None:
            solution = {}
            for e, var in x_vars.items():
                solution[e] = int(round(var.value()))
            cost = sum(self.edge_cost[e] for e, val in solution.items() if val == 1)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = solution
            return

        floor_val = math.floor(val)
        ceil_val = math.ceil(val)
        var_name = x_vars[edge_to_branch].name

        left_model = copy.deepcopy(model)
        left_var_dict = left_model.variablesDict()
        left_x_vars = {}
        for e in self.edges:
            left_x_vars[e] = left_var_dict[x_vars[e].name]
        left_model += (left_x_vars[edge_to_branch] <= floor_val)
        self.branch_and_bound(left_model, left_x_vars, depth + 1)

       
        right_model = copy.deepcopy(model)
        right_var_dict = right_model.variablesDict()
        right_x_vars = {}
        for e in self.edges:
            right_x_vars[e] = right_var_dict[x_vars[e].name]
        right_model += (right_x_vars[edge_to_branch] >= ceil_val)
        self.branch_and_bound(right_model, right_x_vars, depth + 1)

    def solve(self):
        """
        Builds LP relaxation and runs custom Branch and Bound.
        Returns:
          path: list of node indices from source to target
          cost: total cost of the path
        """
      
        if self.source < 0 or self.source >= self.n:
            return None, None
        if self.target < 0 or self.target >= self.n:
            return None, None
        if not self.edges:
            return None, None

        model, x_vars = self.build_lp_relaxation()
        self.best_cost = math.inf
        self.best_solution = None

        self.branch_and_bound(model, x_vars)

        if self.best_solution is None or self.best_cost == math.inf:
            return None, None

       
        chosen_edges = {e for e, val in self.best_solution.items() if val == 1}

        path = [self.source]
        current = self.source
        visited = {self.source}
        max_steps = self.n + 5  

        for _ in range(max_steps):
            if current == self.target:
                return path, self.best_cost

            moved = False
            for (u, v) in chosen_edges:
                if u == current and v not in visited:
                    path.append(v)
                    visited.add(v)
                    current = v
                    moved = True
                    break

            if not moved:
                break

        return None, None



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

    solver = BranchAndBoundILP(adj, start, end)
    path, cost = solver.solve()

    if path is None:
        return jsonify({"path": None, "cost": None, "message": "No path found"}), 200

    return jsonify({"path": path, "cost": cost}), 200


if __name__ == "__main__":
    app.run(debug=True)